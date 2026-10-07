#!/usr/bin/env python3
"""Check the Qwen3 graph builder against Hugging Face transformers, on a small random model.

Two steps, in two environments. `make` (reference venv: torch + transformers) writes a small
random Qwen3 checkpoint, an input and the hidden state transformers computes for it. `check`
(model-compiler venv, after `clm-compile --onnx` on that checkpoint) runs the generated ONNX
graphs one after another and compares, twice: the text alone, and three texts laid end to end
in one pass, each of which has to come out as it does alone.

    .venv-ref/bin/python tools/clm_verify_onnx.py make build/tiny_qwen
    bin/clm-compile build/tiny_qwen -o build/tiny_qwen/out --seq_lens 32 --layers_per_graph 2 --onnx
    $MODEL_COMPILER_BIN/python tools/clm_verify_onnx.py check build/tiny_qwen --seq_len 32 --layers_per_graph 2

A third step, `real`, does the same for the real encoder: the reference texts of
tools/clm_reference.py, laid end to end in as few passes as they fit, through the real ONNX
graphs (in fp32, before any quantizing), against what transformers embedded each to alone.

    $MODEL_COMPILER_BIN/python tools/clm_verify_onnx.py real models/Qwen3-8B --out build/clm --reference build/clm_reference
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
    # The same tokens as three texts of their own: what each comes out as alone.
    cuts = [0, max(1, tokens // 5), max(2, tokens // 2), tokens]
    with torch.no_grad():
        parts = [model(inputs_embeds=embeds[:, a:b]).last_hidden_state[0].numpy() for a, b in zip(cuts, cuts[1:])]
    np.save(path / "parts.npy", np.concatenate(parts))
    (path / "parts.json").write_text(json.dumps([b - a for a, b in zip(cuts, cuts[1:])]))
    print(f"{path}: {sum(p.numel() for p in model.parameters()):,} parameters, {tokens} tokens")


def check(path: Path, seq_len: int, layers_per_graph: int):
    import onnxruntime
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from clm_sima.cli import graph_layers, model_name
    from clm_sima.config import QwenConfig
    from clm_sima.hostio import packed_mask
    cfg = QwenConfig.from_checkpoint(path)
    embeds, expected = np.load(path / "embeds.npy"), np.load(path / "hidden.npy")
    tokens = len(embeds)
    sessions = [onnxruntime.InferenceSession(str(path / "out" / "onnx_files" / f"{model_name(seq_len, first, count)}.onnx"),
                                             providers=["CPUExecutionProvider"])
                for first, count in graph_layers(cfg, layers_per_graph)]

    def run(lengths: list[int]) -> np.ndarray:
        hidden = np.zeros((1, cfg.hidden_size, 1, seq_len), np.float32)      # padded on the right
        hidden[0, :, 0, :tokens] = embeds.T
        mask = packed_mask(lengths, seq_len)
        for session in sessions:
            hidden, = session.run(None, {"hidden_in": hidden, "mask": mask})
        return hidden[0, :, 0, :tokens].T

    alone = np.abs(run([tokens]) - expected).max()
    print(f"{tokens} tokens in a {seq_len}-token graph: max |onnx - transformers| = {alone:.2e} "
          f"(values up to {np.abs(expected).max():.2f})")
    lengths = json.loads((path / "parts.json").read_text())
    packed = np.abs(run(lengths) - np.load(path / "parts.npy")).max()
    print(f"the same tokens as {len(lengths)} texts of {lengths} tokens in one pass: "
          f"max |onnx - transformers, each alone| = {packed:.2e}")
    sys.exit(0 if max(alone, packed) < 1e-3 else 1)


def real(path: Path, out: Path, reference: Path, seq_len: int, layers_per_graph: int):
    import onnxruntime
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from clm_sima.cli import graph_layers, model_name
    from clm_sima.config import QwenConfig
    from clm_sima.hostio import offsets, packed_mask
    from clm_sima.weights import QwenWeights
    cfg = QwenConfig.from_checkpoint(path)
    texts = json.loads((reference / "cases.json").read_text())["texts"]
    want = np.load(reference / "hidden.npy")
    table = QwenWeights(path).raw("model.embed_tokens.weight")
    # Longest first, each into the first pass it fits: what the board's runtime does.
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]["ids"]))
    passes: list[list[int]] = []
    for i in order:
        for group in passes:
            if sum(len(texts[j]["ids"]) for j in group) + len(texts[i]["ids"]) <= seq_len:
                group.append(i)
                break
        else:
            passes.append([i])
    hidden = np.zeros((len(passes), cfg.hidden_size, 1, seq_len), np.float32)
    masks = []
    for p, group in enumerate(passes):
        lengths = [len(texts[i]["ids"]) for i in group]
        ids = [token for i in group for token in texts[i]["ids"]]
        hidden[p, :, 0, :len(ids)] = table[ids].T
        masks.append(packed_mask(lengths, seq_len))
    for first, count in graph_layers(cfg, layers_per_graph):     # a graph at a time: each is gigabytes
        session = onnxruntime.InferenceSession(str(out / "onnx_files" / f"{model_name(seq_len, first, count)}.onnx"),
                                               providers=["CPUExecutionProvider"])
        for p in range(len(passes)):
            hidden[p] = session.run(None, {"hidden_in": hidden[p:p + 1], "mask": masks[p]})[0][0]
        del session
        print(f"layers {first}-{first + count - 1} done", flush=True)
    worst = 1.0
    for p, group in enumerate(passes):
        lengths = [len(texts[i]["ids"]) for i in group]
        for i, start, length in zip(group, offsets(lengths), lengths):
            got = hidden[p, :, 0, start + length - 1]
            worst = min(worst, float(got @ want[i] / np.linalg.norm(got) / np.linalg.norm(want[i])))
    print(f"{len(texts)} reference texts in {len(passes)} passes of {seq_len}: worst cosine with transformers, each alone, {worst:.6f}")
    sys.exit(0 if worst > 0.9999 else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["make", "check", "real"])
    ap.add_argument("path", type=Path)
    ap.add_argument("--tokens", type=int, default=21)
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--layers_per_graph", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=256, help="make: hidden size of the random model")
    ap.add_argument("--head_dim", type=int, default=32, help="make: size of one attention head")
    ap.add_argument("--layers", type=int, default=4, help="make: number of decoder layers")
    ap.add_argument("--out", type=Path, help="real: the clm-compile output directory")
    ap.add_argument("--reference", type=Path, default=Path("build/clm_reference"), help="real: the PyTorch dump")
    args = ap.parse_args()
    if args.step == "real":
        real(args.path, args.out, args.reference, args.seq_len, args.layers_per_graph)
    make(args.path, args.tokens, args.hidden, args.head_dim, args.layers) if args.step == "make" else check(args.path, args.seq_len, args.layers_per_graph)
