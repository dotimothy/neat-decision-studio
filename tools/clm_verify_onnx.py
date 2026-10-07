#!/usr/bin/env python3
"""Check the Qwen3 graph builder against Hugging Face transformers, on a small random model.

Two steps, in two environments. `make` (reference venv: torch + transformers) writes a small
random Qwen3 checkpoint, an input and the hidden state transformers computes for it. `check`
(model-compiler venv, after `clm-compile --onnx` on that checkpoint) runs the generated ONNX
graphs one after another and compares.

    .venv-ref/bin/python tools/clm_verify_onnx.py make build/tiny_qwen
    bin/clm-compile build/tiny_qwen -o build/tiny_qwen/out --seq_len 32 --layers_per_graph 2 --onnx
    $MODEL_COMPILER_BIN/python tools/clm_verify_onnx.py check build/tiny_qwen --seq_len 32 --layers_per_graph 2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def make(path: Path, tokens: int, hidden: int, head_dim: int, layers: int):
    import torch
    from transformers import Qwen3Config, Qwen3Model
    torch.manual_seed(7)
    config = Qwen3Config(hidden_size=hidden, intermediate_size=2 * hidden, num_attention_heads=8, num_key_value_heads=2,
                         head_dim=head_dim, num_hidden_layers=layers, vocab_size=1000, rope_theta=1000000.0,
                         rms_norm_eps=1e-6, max_position_embeddings=512)
    model = Qwen3Model(config).eval()
    with torch.no_grad():                    # norm weights of 1 would hide a missing one
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight") or "layernorm" in name:
                parameter.copy_(1.0 + 0.3 * torch.randn_like(parameter))
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    raw = json.loads((path / "config.json").read_text())
    raw["model_type"] = "qwen3"
    (path / "config.json").write_text(json.dumps(raw, indent=1))
    embeds = torch.randn(1, tokens, config.hidden_size)
    with torch.no_grad():
        hidden = model(inputs_embeds=embeds).last_hidden_state
    np.save(path / "embeds.npy", embeds[0].numpy())
    np.save(path / "hidden.npy", hidden[0].numpy())
    print(f"{path}: {sum(p.numel() for p in model.parameters()):,} parameters, {tokens} tokens")


def check(path: Path, seq_len: int, layers_per_graph: int):
    import onnxruntime
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from clm_sima.cli import graph_layers, model_name
    from clm_sima.config import QwenConfig
    cfg = QwenConfig.from_checkpoint(path)
    embeds, expected = np.load(path / "embeds.npy"), np.load(path / "hidden.npy")
    tokens = len(embeds)
    hidden = np.zeros((1, cfg.hidden_size, 1, seq_len), np.float32)      # padded on the right
    hidden[0, :, 0, :tokens] = embeds.T
    for first, count in graph_layers(cfg, layers_per_graph):
        onnx_file = path / "out" / "onnx_files" / f"{model_name(seq_len, first, count)}.onnx"
        session = onnxruntime.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        hidden, = session.run(None, {"hidden_in": hidden})
    got = hidden[0, :, 0, :tokens].T
    error = np.abs(got - expected).max()
    print(f"{tokens} tokens in a {seq_len}-token graph: max |onnx - transformers| = {error:.2e} "
          f"(values up to {np.abs(expected).max():.2f})")
    sys.exit(0 if error < 1e-3 else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["make", "check"])
    ap.add_argument("path", type=Path)
    ap.add_argument("--tokens", type=int, default=21)
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--layers_per_graph", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=256, help="make: hidden size of the random model")
    ap.add_argument("--head_dim", type=int, default=32, help="make: size of one attention head")
    ap.add_argument("--layers", type=int, default=4, help="make: number of decoder layers")
    args = ap.parse_args()
    make(args.path, args.tokens, args.hidden, args.head_dim, args.layers) if args.step == "make" else check(args.path, args.seq_len, args.layers_per_graph)
