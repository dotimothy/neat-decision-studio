"""d1-compile: LiquidAI's d1 models -> MLA ELFs, in the passes `llima-compile` uses.

    d1-compile models/d1-omni-600m -o build/d1-omni --seq_lens 128,512
    d1-compile models/d1-3b -o build/d1-3b --seq_lens 128,512 --layers_per_graph 4

Three kinds of graph come out. The trunk's layers and the decision head are a chain of graphs
of `--layers_per_graph` layers a sequence length (all of them in one graph by default: the
model is small). The vision tower is one graph for up to `--patches` patches of a picture, and
the projector one for a quarter as many. Each sequence length has its own copy of the trunk's
weights: the board loads the ones it is told to, and uses the shortest a pass fits in.

The pass flags (`--onnx`, `--quantize`, `--compile`, `--devkit`) run one pass and mean what
they mean for `llima-compile`; with none of them, all four run. `--only trunk` (or `vision`,
`projector`) restricts a run to some of the graphs, and `--no-vision` leaves pictures out.
"""
import argparse
import json
import logging
import shutil
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

from sima_lmm.model.base import FileGenMode, FileGenPrecision, LoraGenMode

from d1_sima import hostio, lmio
from d1_sima.config import D1Config
from d1_sima.model import D1ProjectorModel, D1TrunkModel, D1VisionModel
from d1_sima.weights import D1Weights

_PASSES = [
    ("onnx", FileGenMode.SOURCE_TO_ONNX),
    ("quantize", FileGenMode.ONNX_TO_QUANT),
    ("compile", FileGenMode.MODEL_SDK_COMPILE),
]
_LOG_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING,
               "ERROR": logging.ERROR}


def graph_layers(cfg: D1Config, layers_per_graph: int) -> list[tuple[int, int]]:
    """(first layer, number of layers) of every trunk graph, in the order they run."""
    return [(first, min(layers_per_graph, cfg.num_hidden_layers - first))
            for first in range(0, cfg.num_hidden_layers, layers_per_graph)]


def trunk_name(seq_len: int, first: int, count: int) -> str:
    return f"d1_s{seq_len}_l{first:02d}n{count}"


def vision_name(patches: int) -> str:
    return f"d1_vision_n{patches}"


def projector_name(patches: int) -> str:
    return f"d1_projector_m{patches // hostio.MERGE ** 2}"


def build_models(cfg: D1Config, model_path: Path, output: Path, seq_lens: list[int], graphs: list[tuple[int, int]],
                 patches: int, only: set[str]) -> list:
    common = {"onnx_path": output / "onnx_files", "sima_path": output / "sima_files", "model_path": model_path}
    models = []
    if "trunk" in only:
        models += [D1TrunkModel(cfg, trunk_name(s, first, count), seq_len=s, first_layer=first, num_layers=count, **common)
                   for s in seq_lens for first, count in graphs]
    if patches and "vision" in only:
        models.append(D1VisionModel(cfg, vision_name(patches), seq_len=patches, **common))
    if patches and "projector" in only:
        models.append(D1ProjectorModel(cfg, projector_name(patches), seq_len=patches // hostio.MERGE ** 2, **common))
    return models


def bfloat16(values: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bits, rounded to the nearest even: the MLA's activation type, so that
    looking a token up on the board is a memcpy."""
    bits = np.ascontiguousarray(values, np.float32).view(np.uint32).astype(np.uint64)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def extract_elf(output: Path, devkit: Path, name: str) -> tuple[str, list[str]]:
    """A graph's ELF, copied to the devkit directory, and the inputs the MLA stage takes, in its
    order. That is the order the graph first uses them in, which need not be the order they
    were declared in: a trunk that starts with a convolution takes the taps' masks before the
    attention mask. The package's description says which."""
    mpk = output / "sima_files" / "mpk" / f"{name}_mpk.tar.gz"
    elf_name = f"{name}_stage1_mla.elf"
    with tarfile.open(mpk) as tar:
        with tar.extractfile(f"{name}_mpk.json") as described:
            plugins = json.load(described)["plugins"]
        feeds = {out["name"]: [node["name"] for node in plugin["input_nodes"]] for plugin in plugins for out in plugin["output_nodes"]}
        stages = [plugin for plugin in plugins if plugin["processor"] == "MLA"]
        if len(stages) != 1:
            raise RuntimeError(f"{mpk} describes {len(stages)} MLA stages; a graph must compile to exactly one")
        # Each of the stage's inputs is a graph input, or a cast of one.
        inputs = [feeds[node["name"]][0] if node["name"] in feeds else node["name"] for node in stages[0]["input_nodes"]]
        members = [m for m in tar.getmembers() if m.name.endswith(".elf")]
        if len(members) != 1:
            raise RuntimeError(
                f"{mpk} holds {len(members)} ELF files; a graph must compile to exactly one MLA stage. "
                "Check the compile log for nodes that were not assigned to the MLA."
            )
        with tar.extractfile(members[0]) as src, open(devkit / elf_name, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return elf_name, inputs


def gen_devkit_files(cfg: D1Config, model_path: Path, output: Path, seq_lens: list[int],
                     graphs: list[tuple[int, int]], patches: int, precision: FileGenPrecision):
    """Everything the board needs, in one directory: ELFs, tokenizer, embeddings, config."""
    from tokenizers import Tokenizer

    devkit = output / "sima_files" / "devkit"
    devkit.mkdir(parents=True, exist_ok=True)
    chains, inputs = {}, None
    for s in seq_lens:
        taken = [extract_elf(output, devkit, trunk_name(s, first, count)) for first, count in graphs]
        chains[str(s)] = [elf for elf, _ in taken]
        # What each graph of a chain takes, in the MLA's order: the same for every length.
        if inputs not in (None, [order for _, order in taken]):
            raise RuntimeError(f"the {s}-token chain takes its inputs in another order than the shorter ones")
        inputs = [order for _, order in taken]
    for (first, count), order in zip(graphs, inputs):
        declared = D1TrunkModel(cfg, "inputs", onnx_path=output, sima_path=output, seq_len=16, model_path=model_path,
                                first_layer=first, num_layers=count).input_names
        if sorted(order) != sorted(declared):
            raise RuntimeError(f"the graph of layers {first}..{first + count - 1} takes {order}, not {declared}")
    vision = None
    if patches:
        tower, tower_inputs = extract_elf(output, devkit, vision_name(patches))
        projector, projector_inputs = extract_elf(output, devkit, projector_name(patches))
        if sorted(tower_inputs) != ["mask", "patches", "positions"]:
            raise RuntimeError(f"the vision tower takes {tower_inputs}")
        vision = {"patches": patches, "elf": tower, "inputs": tower_inputs, "projector": projector,
                  "projector_inputs": projector_inputs,
                  "patch_size": cfg.patch_size, "merge": cfg.downsample, "hidden_size": cfg.vision_hidden_size,
                  "min_pixels": hostio.MIN_PIXELS, "max_pixels": hostio.MAX_PIXELS,
                  "position_embedding": "position_embedding.f32", "position_grid": cfg.position_grid}
    wanted = {name for elfs in chains.values() for name in elfs} | ({vision["elf"], vision["projector"]} if vision else set())
    for stale in devkit.glob("*_stage1_mla.elf"):
        if stale.name not in wanted:
            stale.unlink()

    weights = D1Weights(model_path, cfg.head_dim)
    bfloat16(weights.raw(f"{cfg.trunk}.embed_tokens.weight")).tofile(devkit / "token_embeddings.bf16")
    if cfg.family == "omni":
        weights.raw("head.type_emb.weight").astype(np.float32).tofile(devkit / "type_embeddings.f32")
    if vision:
        weights.raw(f"{cfg.tower}.embeddings.position_embedding.weight").astype(np.float32).tofile(
            devkit / "position_embedding.f32")
    shutil.copy(model_path / "tokenizer.json", devkit / "tokenizer.json")
    for licence in ("LICENSE",):             # the model's licence travels with its weights
        if (model_path / licence).exists():
            shutil.copy(model_path / licence, devkit / licence)

    raw = json.loads((model_path / "config.json").read_text())
    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    if cfg.family == "lm":
        # A causal language model: its answer is read off the next token's logits, which are
        # the final hidden state against the rows of the embedding table (the two are tied).
        tokens = {name: tokenizer.token_to_id(token) for name, token in (
            ("im_start", lmio.IM_START), ("im_end", lmio.IM_END), ("image", lmio.IMAGE), ("image_start", lmio.IMAGE_START),
            ("image_end", lmio.IMAGE_END))}
        if any(value is None for value in tokens.values()) or not raw.get("tie_word_embeddings"):
            raise RuntimeError(f"the tokenizer lacks a special token ({tokens}), or the embeddings are not tied")
        if vision:
            vision["resample"] = "bicubic"       # LFM2-VL's processor; d1-omni's own code resizes bilinearly
        config = {
            "kind": "d1", "family": "lm", "model": "d1-3B", "precision": precision.value,
            "hidden_size": cfg.hidden_size, "vocab_size": cfg.vocab_size, "num_hidden_layers": cfg.num_hidden_layers,
            "elfs": chains, "inputs": inputs, "vision": vision,
            "token_embeddings": "token_embeddings.bf16", "tokenizer": "tokenizer.json",
            "tokens": {"bos": raw["text_config"]["bos_token_id"], **tokens},
            "readout": lmio.readout_table(lambda text: tokenizer.encode(text, add_special_tokens=False).ids),
            "licence": "LFM Open License v1.0 (LICENSE)",
        }
        (devkit / "d1_config.json").write_text(json.dumps(config, indent=4))
        return
    tokens = {name: tokenizer.token_to_id(token) for name, token in hostio.DELIMITERS.items()}
    if any(value is None for value in tokens.values()):
        raise RuntimeError(f"the tokenizer lacks a delimiter: {tokens}")
    config = {
        "kind": "d1",
        "family": "omni",
        "model": "d1-omni-600M",
        "precision": precision.value,
        "hidden_size": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        # Sequence length -> its chain of graphs, in the order they run, and what each takes.
        # The last graph gives the scorer's output at every position.
        "elfs": chains,
        "inputs": inputs,
        "vision": vision,
        "token_embeddings": "token_embeddings.bf16",
        "type_embeddings": "type_embeddings.f32",
        "question_types": hostio.QTYPES,
        "tokenizer": "tokenizer.json",
        "tokens": {"bos": raw["bos_token_id"], **tokens},
        "temperatures": raw.get("temperatures", {}),
        "text_limit": raw["max_length"],
        "image_text_limit": raw["image_text_length"],
        "licence": "LFM Open License v1.0 (LICENSE)",
    }
    (devkit / "d1_config.json").write_text(json.dumps(config, indent=4))


def main():
    parser = argparse.ArgumentParser(
        prog="d1-compile", description="LiquidAI d1 compiler for SiMa.ai Modalix"
    )
    parser.add_argument("model_path", type=Path, help="d1-omni-600M or d1-3B checkpoint directory")
    parser.add_argument("-o", "--output", type=Path, help="Output directory (default: model name)")
    parser.add_argument("--log_level", default="WARNING", choices=_LOG_LEVELS, metavar="LEVEL")
    group = parser.add_argument_group("Options to run only one compiler pass")
    egroup = group.add_mutually_exclusive_group()
    egroup.add_argument("--onnx", action="store_true", help="Compile to ONNX files")
    egroup.add_argument("--quantize", action="store_true",
                        help="Convert ONNX files to Model SDK files and quantize them")
    egroup.add_argument("--compile", action="store_true",
                        help="Compile quantized Model SDK files to machine code")
    egroup.add_argument("--devkit", action="store_true",
                        help="Produce the directory that is deployed to the DevKit")
    group = parser.add_argument_group("Model compilation parameters")
    group.add_argument("--seq_lens", default="128,512", metavar="LIST",
                       help="Comma-separated token counts to compile the trunk for (default: %(default)s)")
    group.add_argument("--layers_per_graph", type=int, default=16,
                       help="Trunk layers in one graph (default: %(default)s, all of them)")
    group.add_argument("--patches", type=int, default=hostio.MAX_PATCHES,
                       help="Patches of a picture the vision tower is compiled for (default: %(default)s, "
                            "a 512 x 512 picture's)")
    group.add_argument("--no-vision", action="store_true", help="Leave the vision tower out: text questions only")
    group.add_argument("--precision", default=FileGenPrecision.A_BF16_W_INT8.value,
                       choices=[p.value for p in FileGenPrecision],
                       help="Activations are always BF16; this selects the weight precision "
                            "(default: %(default)s)")
    group = parser.add_argument_group("Advanced options")
    group.add_argument("--only", default="trunk,vision,projector", metavar="LIST",
                       help="Which graphs to build: trunk, vision, projector (default: all)")
    group.add_argument("--graphs", metavar="LIST",
                       help="Comma-separated indices of the trunk graphs to build (default: all), to build "
                            "a long chain in several runs at once")
    group.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False,
                       help="Only compile files that are missing")
    args = parser.parse_args()

    output = args.output or Path(args.model_path.name)
    cfg = D1Config.from_checkpoint(args.model_path)
    seq_lens = sorted({int(s) for s in args.seq_lens.split(",")})
    for seq_len in seq_lens:
        if seq_len % 16 or seq_len < 16:
            sys.exit(f"d1-compile: seq_len {seq_len} must be a multiple of 16")
    patches = 0 if args.no_vision else args.patches
    if patches % 64:
        sys.exit(f"d1-compile: --patches {patches} must be a multiple of 64")
    graphs = graph_layers(cfg, args.layers_per_graph)
    only = set(args.only.split(","))
    precision = FileGenPrecision(args.precision)

    selected = [name for name in ("onnx", "quantize", "compile", "devkit") if getattr(args, name)]
    layer_cfg = {"precision": precision, "lora": LoraGenMode.LORA_DISABLED}
    for name, mode in _PASSES:
        if selected and selected != [name]:
            continue
        chosen = graphs if not args.graphs else [graphs[int(i)] for i in args.graphs.split(",")]
        for model in build_models(cfg, args.model_path, output, seq_lens, chosen, patches, only):
            start = time.time()
            created = model.gen_files(
                mode, layer_cfg=layer_cfg, log_level=_LOG_LEVELS[args.log_level], resume=args.resume
            )
            print(f"[{name}] {model.get_gen_file_name(mode)} "
                  f"{'created' if created else 'kept'} in {time.time() - start:.0f}s", flush=True)
    if (not selected or selected == ["devkit"]) and only == {"trunk", "vision", "projector"} and not args.graphs:
        gen_devkit_files(cfg, args.model_path, output, seq_lens, graphs, patches, precision)
        print(f"[devkit] {output / 'sima_files' / 'devkit'}", flush=True)


if __name__ == "__main__":
    main()
