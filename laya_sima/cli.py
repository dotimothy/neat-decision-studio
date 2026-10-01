"""laya-compile: Laya checkpoint -> MLA ELF, in the passes `llima-compile` uses.

    laya-compile models/laya -o build/laya --seq_lens 128,512

Each sequence length is its own graph and its own ELF; the runtime picks the smallest one a
request fits in. The pass flags (`--onnx`, `--quantize`, `--compile`, `--devkit`) run one pass
and mean what they mean for `llima-compile`; with none of them, all four run.
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
from ml_dtypes import bfloat16

from sima_lmm.model.base import FileGenMode, FileGenPrecision, LoraGenMode

from laya_sima.config import LayaConfig
from laya_sima.model import LayaModel
from laya_sima.weights import LayaWeights

_PASSES = [
    ("onnx", FileGenMode.SOURCE_TO_ONNX),
    ("quantize", FileGenMode.ONNX_TO_QUANT),
    ("compile", FileGenMode.MODEL_SDK_COMPILE),
]
_LOG_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING,
               "ERROR": logging.ERROR}


def model_name(seq_len: int, encoder_only: bool = False) -> str:
    return f"laya_enc_s{seq_len}" if encoder_only else f"laya_s{seq_len}"


def build_model(cfg: LayaConfig, model_path: Path, output: Path, seq_len: int,
                encoder_only: bool = False, int8_weights: str = "none") -> LayaModel:
    return LayaModel(
        cfg, model_name(seq_len, encoder_only), onnx_path=output / "onnx_files",
        sima_path=output / "sima_files", seq_len=seq_len, model_path=model_path,
        encoder_only=encoder_only, int8_weights=int8_weights
    )


def gen_devkit_files(cfg: LayaConfig, model_path: Path, output: Path, seq_lens: list[int],
                     precision: FileGenPrecision, int8_weights: str = "none"):
    args_model_name = model_path.name
    """Everything the board needs, in one directory: ELFs, tokenizer, CPU-side tensors, config."""
    devkit = output / "sima_files" / "devkit"
    devkit.mkdir(parents=True, exist_ok=True)
    weights = LayaWeights(model_path)

    elfs = {}
    for seq_len in seq_lens:
        mpk = output / "sima_files" / "mpk" / f"{model_name(seq_len)}_mpk.tar.gz"
        with tarfile.open(mpk) as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".elf")]
            if len(members) != 1:
                raise RuntimeError(
                    f"{mpk} holds {len(members)} ELF files; Laya must compile to exactly one MLA "
                    "stage. Check the compile log for nodes that were not assigned to the MLA."
                )
            elf_name = f"{model_name(seq_len)}_stage1_mla.elf"
            with tar.extractfile(members[0]) as src, open(devkit / elf_name, "wb") as dst:
                shutil.copyfileobj(src, dst)
        elfs[str(seq_len)] = elf_name

    # A previous run may have packaged sequence lengths that are no longer wanted.
    for stale in devkit.glob("*_stage1_mla.elf"):
        if stale.name not in elfs.values():
            stale.unlink()

    # Token embeddings, already in the MLA's activation type: a lookup is then a memcpy.
    weights.raw("encoder.embeddings.tok_embeddings.weight").astype(bfloat16).tofile(
        devkit / "token_embeddings.bf16"
    )
    # The act head beyond what the graph computes, as float32: w_feats (256x4), w_out (2x256),
    # b_out (2), concatenated in that order.
    act0 = weights.raw("act_head.0.weight")
    np.concatenate([
        act0[:, cfg.hidden_size:].ravel(), weights.raw("act_head.2.weight").ravel(),
        weights.raw("act_head.2.bias").ravel()
    ]).astype(np.float32).tofile(devkit / "act_tail.f32")
    shutil.copy(model_path / "tokenizer" / "tokenizer.json", devkit / "tokenizer.json")

    config = {
        "model": args_model_name,
        "precision": precision.value if int8_weights == "none" else f"BF16 + int8 {int8_weights}",
        "hidden_size": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
        "act_hidden_size": cfg.act_hidden_size,
        "num_qtypes": cfg.num_qtypes,
        "qtype_channels": cfg.qtype_channels,
        "sliding_window": cfg.sliding_window,
        "pad_token_id": cfg.pad_token_id,
        "cls_token_id": cfg.cls_token_id,
        "sep_token_id": cfg.sep_token_id,
        "mask_token_id": cfg.mask_token_id,
        "mask_token": cfg.mask_token,
        "max_len": cfg.max_len,
        "head_max_len": cfg.head_max_len,
        "temperature": cfg.temperature,
        "temperature_by_options": cfg.temperature_by_options,
        "elfs": elfs,
        "token_embeddings": "token_embeddings.bf16",
        "act_tail": "act_tail.f32",
        "tokenizer": "tokenizer.json",
    }
    (devkit / "laya_config.json").write_text(json.dumps(config, indent=4))


def main():
    parser = argparse.ArgumentParser(
        prog="laya-compile", description="Laya decision model compiler for SiMa.ai Modalix"
    )
    parser.add_argument("model_path", type=Path, help="Laya checkpoint directory")
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
                       help="Comma-separated token counts to compile a graph for (default: %(default)s)")
    group.add_argument("--precision", default=FileGenPrecision.BF16.value,
                       choices=[p.value for p in FileGenPrecision],
                       help="Activations are always BF16; this selects the weight precision")
    group.add_argument("--int8_weights", default="none", choices=["none", "mlp", "encoder"],
                       help="With --precision BF16: also store these encoder weights as int8, in "
                            "the same graph. mlp = the MLP matrices of every encoder layer, "
                            "encoder = all encoder matrices. The decision head stays as it is.")
    group = parser.add_argument_group("Advanced options")
    group.add_argument("--encoder_only", action="store_true",
                       help="Diagnostic: compile only the encoder, with its hidden state as the "
                            "output (see tools/train_dino.py). No devkit directory is produced.")
    group.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False,
                       help="Only compile files that are missing")
    args = parser.parse_args()

    output = args.output or Path(args.model_path.name)
    seq_lens = sorted({int(s) for s in args.seq_lens.split(",")})
    precision = FileGenPrecision(args.precision)
    cfg = LayaConfig.from_checkpoint(args.model_path)
    for seq_len in seq_lens:
        if seq_len % 16 or not 16 <= seq_len <= cfg.max_len:
            sys.exit(f"laya-compile: seq_len {seq_len} must be a multiple of 16 in [16, {cfg.max_len}]")

    selected = [name for name in ("onnx", "quantize", "compile", "devkit") if getattr(args, name)]
    layer_cfg = {"precision": precision, "lora": LoraGenMode.LORA_DISABLED}
    for name, mode in _PASSES:
        if selected and selected != [name]:
            continue
        for seq_len in seq_lens:
            model = build_model(cfg, args.model_path, output, seq_len, args.encoder_only,
                                args.int8_weights)
            start = time.time()
            created = model.gen_files(
                mode, layer_cfg=layer_cfg, log_level=_LOG_LEVELS[args.log_level], resume=args.resume
            )
            print(f"[{name}] {model.get_gen_file_name(mode)} "
                  f"{'created' if created else 'kept'} in {time.time() - start:.0f}s", flush=True)
    if args.encoder_only:
        return
    if not selected or selected == ["devkit"]:
        gen_devkit_files(cfg, args.model_path, output, seq_lens, precision, args.int8_weights)
        print(f"[devkit] {output / 'sima_files' / 'devkit'}", flush=True)


if __name__ == "__main__":
    main()
