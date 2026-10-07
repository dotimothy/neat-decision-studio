#!/usr/bin/env python3
"""A small made-up CLM model directory for the mock build of the runtime (see README):
random embeddings and heads, and graph files that are only names."""
import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(3)
hidden, vocab, width, proj = 64, 256, 48, 16
bits = rng.standard_normal((vocab, hidden)).astype(np.float32).view(np.uint32) >> 16
bits.astype(np.uint16).tofile(out / "token_embeddings.bf16")
chunks, heads = [], {}
for head in ("state_head", "action_head"):
    tensors = [("inp.weight", (width, hidden)), ("inp.bias", (width,)), ("hidden.0.weight", (width, width)), ("hidden.0.bias", (width,)),
               ("norms.0.weight", (width,)), ("norms.0.bias", (width,)), ("out.weight", (proj, width)), ("out.bias", (proj,))]
    heads[head] = [{"name": name, "shape": list(shape)} for name, shape in tensors]
    chunks += [(np.ones(shape) if name == "norms.0.weight" else rng.standard_normal(shape) * 0.3).astype(np.float32).ravel() for name, shape in tensors]
np.concatenate(chunks).tofile(out / "heads.f32")
(out / "heads.json").write_text(json.dumps({"hidden_size": hidden, "width": width, "depth": 3, "projection_dim": proj, "activation": "gelu",
                                            "layernorm": True, "residual": False, "scale": 20.0, "heads": heads}))
(out / "tokenizer.json").write_text("{}")
chains = {str(s): [f"mock_s{s}_l{i:02d}n2_stage1_mla.elf" for i in (0, 2, 4)] for s in (32, 128)}
for files in chains.values():
    for name in files:
        (out / name).write_bytes(b"mock")
(out / "clm_config.json").write_text(json.dumps({"kind": "clm", "model": "CLM-v0.1-8B", "encoder": "mock", "precision": "A_BF16_W_INT8",
    "hidden_size": hidden, "vocab_size": vocab, "num_hidden_layers": 6, "elfs": chains,
    "token_embeddings": "token_embeddings.bf16", "tokenizer": "tokenizer.json", "heads": "heads.f32"}, indent=1))
print(out)
