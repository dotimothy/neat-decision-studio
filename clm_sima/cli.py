"""clm-compile: Qwen3-8B (CLM's encoder) -> MLA ELFs, in the passes `llima-compile` uses.

    clm-compile models/Qwen3-8B -o build/clm --seq_len 128 --layers_per_graph 4

The decoder's layers are cut into consecutive graphs of `--layers_per_graph` layers, each its
own ELF; the board runs them in order. The pass flags (`--onnx`, `--quantize`, `--compile`,
`--devkit`) run one pass and mean what they mean for `llima-compile`; with none of them, all
four run. `--graphs 0,1` restricts a run to some of the graphs, to try one before building all.
"""
import argparse
import json
import logging
import shutil
import sys
import tarfile
import time
from pathlib import Path

from sima_lmm.model.base import FileGenMode, FileGenPrecision, LoraGenMode

from clm_sima.config import QwenConfig
from clm_sima.model import QwenLayersModel
from clm_sima.weights import QwenWeights

_PASSES = [
    ("onnx", FileGenMode.SOURCE_TO_ONNX),
    ("quantize", FileGenMode.ONNX_TO_QUANT),
    ("compile", FileGenMode.MODEL_SDK_COMPILE),
]
_LOG_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING,
               "ERROR": logging.ERROR}


def graph_layers(cfg: QwenConfig, layers_per_graph: int) -> list[tuple[int, int]]:
    """(first layer, number of layers) of every graph, in the order they run."""
    return [(first, min(layers_per_graph, cfg.num_hidden_layers - first))
            for first in range(0, cfg.num_hidden_layers, layers_per_graph)]


def model_name(seq_len: int, first: int, count: int) -> str:
    return f"qwen_s{seq_len}_l{first:02d}n{count}"


def build_model(cfg: QwenConfig, model_path: Path, output: Path, seq_len: int, first: int, count: int) -> QwenLayersModel:
    return QwenLayersModel(
        cfg, model_name(seq_len, first, count), onnx_path=output / "onnx_files",
        sima_path=output / "sima_files", seq_len=seq_len, model_path=model_path,
        first_layer=first, num_layers=count
    )


def gen_devkit_files(cfg: QwenConfig, model_path: Path, output: Path, seq_len: int,
                     graphs: list[tuple[int, int]], precision: FileGenPrecision, heads: Path | None):
    """Everything the board needs, in one directory: ELFs, tokenizer, token embeddings, config."""
    devkit = output / "sima_files" / "devkit"
    devkit.mkdir(parents=True, exist_ok=True)

    elfs = []
    for first, count in graphs:
        name = model_name(seq_len, first, count)
        mpk = output / "sima_files" / "mpk" / f"{name}_mpk.tar.gz"
        with tarfile.open(mpk) as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".elf")]
            if len(members) != 1:
                raise RuntimeError(
                    f"{mpk} holds {len(members)} ELF files; a run of layers must compile to exactly "
                    "one MLA stage. Check the compile log for nodes that were not assigned to the MLA."
                )
            elf_name = f"{name}_stage1_mla.elf"
            with tar.extractfile(members[0]) as src, open(devkit / elf_name, "wb") as dst:
                shutil.copyfileobj(src, dst)
        elfs.append(elf_name)
    for stale in devkit.glob("*_stage1_mla.elf"):
        if stale.name not in elfs:
            stale.unlink()

    # Token embeddings as the checkpoint has them, bfloat16: the MLA's activation type, so a
    # lookup is a memcpy.
    QwenWeights(model_path).bf16("model.embed_tokens.weight").tofile(devkit / "token_embeddings.bf16")
    shutil.copy(model_path / "tokenizer.json", devkit / "tokenizer.json")
    if heads is not None:                    # written by tools/clm_heads.py, which needs torch
        for name in ("heads.f32", "heads.json"):
            shutil.copy(heads / name, devkit / name)

    config = {
        "kind": "clm",
        "model": "CLM-v0.1-8B",
        "encoder": model_path.name,
        "precision": precision.value,
        "hidden_size": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "seq_len": seq_len,
        "elfs": elfs,
        "token_embeddings": "token_embeddings.bf16",
        "tokenizer": "tokenizer.json",
        "heads": "heads.f32" if heads is not None else None,
    }
    (devkit / "clm_config.json").write_text(json.dumps(config, indent=4))


def main():
    parser = argparse.ArgumentParser(
        prog="clm-compile", description="Qwen3 encoder compiler for CLM on SiMa.ai Modalix"
    )
    parser.add_argument("model_path", type=Path, help="Qwen3 checkpoint directory")
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
    group.add_argument("--seq_len", type=int, default=128,
                       help="Token positions the graphs are compiled for (default: %(default)s)")
    group.add_argument("--layers_per_graph", type=int, default=4,
                       help="Decoder layers in one graph (default: %(default)s)")
    group.add_argument("--precision", default=FileGenPrecision.A_BF16_W_INT8.value,
                       choices=[p.value for p in FileGenPrecision],
                       help="Activations are always BF16; this selects the weight precision "
                            "(default: %(default)s)")
    group.add_argument("--heads", type=Path,
                       help="Directory with heads.f32 and heads.json (tools/clm_heads.py), for --devkit")
    group = parser.add_argument_group("Advanced options")
    group.add_argument("--graphs", metavar="LIST",
                       help="Comma-separated indices of the graphs to build (default: all)")
    group.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False,
                       help="Only compile files that are missing")
    args = parser.parse_args()

    output = args.output or Path(args.model_path.name)
    cfg = QwenConfig.from_checkpoint(args.model_path)
    if args.seq_len % 16 or args.seq_len < 16:
        sys.exit(f"clm-compile: seq_len {args.seq_len} must be a multiple of 16")
    graphs = graph_layers(cfg, args.layers_per_graph)
    chosen = graphs if not args.graphs else [graphs[int(i)] for i in args.graphs.split(",")]
    precision = FileGenPrecision(args.precision)

    selected = [name for name in ("onnx", "quantize", "compile", "devkit") if getattr(args, name)]
    layer_cfg = {"precision": precision, "lora": LoraGenMode.LORA_DISABLED}
    for name, mode in _PASSES:
        if selected and selected != [name]:
            continue
        for first, count in chosen:
            model = build_model(cfg, args.model_path, output, args.seq_len, first, count)
            start = time.time()
            created = model.gen_files(
                mode, layer_cfg=layer_cfg, log_level=_LOG_LEVELS[args.log_level], resume=args.resume
            )
            print(f"[{name}] {model.get_gen_file_name(mode)} "
                  f"{'created' if created else 'kept'} in {time.time() - start:.0f}s", flush=True)
    if (not selected or selected == ["devkit"]) and not args.graphs:
        gen_devkit_files(cfg, args.model_path, output, args.seq_len, graphs, precision, args.heads)
        print(f"[devkit] {output / 'sima_files' / 'devkit'}", flush=True)


if __name__ == "__main__":
    main()
