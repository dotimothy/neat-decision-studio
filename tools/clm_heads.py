#!/usr/bin/env python3
"""CLM's two projection heads, out of their PyTorch checkpoint and into what the board reads.

The checkpoint (`CLM_v0.1-8B.pt`) is a pickled dict that needs torch to open. The board's
runtime has no torch, so this writes the heads as one flat float32 file and a JSON description
of what is in it, in order:

    heads.json   width, depth, projection size, activation, layernorm, residual, the scale of
                 the scores (exp of the checkpoint's logit_scale), and for each head the
                 tensors in the order they sit in heads.f32, with their shapes
    heads.f32    those tensors, float32, one after another

    .venv-ref/bin/python tools/clm_heads.py models/CLM-v0.1-8B/CLM_v0.1-8B.pt build/clm_heads
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    cfg = dict(ck["cfg"])
    depth = cfg["depth"]
    description = {
        "hidden_size": cfg.get("hidden_size", 4096), "width": cfg["width"], "depth": depth,
        "projection_dim": ck.get("projection_dim", cfg.get("projection_dim", 512)),
        "activation": cfg.get("activation", "gelu"), "layernorm": bool(cfg.get("layernorm", False)),
        "residual": bool(cfg.get("residual", False)),
        "scale": float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100.0)),
        "heads": {},
    }
    chunks = []
    for head in ("state_head", "action_head"):
        state = {k: v.float().numpy() for k, v in ck[head].items()}
        names = ["inp.weight", "inp.bias"]
        for i in range(depth - 2):
            names += [f"hidden.{i}.weight", f"hidden.{i}.bias"]
            if description["layernorm"]:
                names += [f"norms.{i}.weight", f"norms.{i}.bias"]
        names += ["out.weight", "out.bias"]
        unused = sorted(set(state) - set(names))
        if unused:
            raise SystemExit(f"{head}: tensors this layout does not know: {unused}")
        description["heads"][head] = [{"name": n, "shape": list(state[n].shape)} for n in names]
        chunks += [state[n].ravel() for n in names]
    args.out.mkdir(parents=True, exist_ok=True)
    flat = np.concatenate(chunks).astype(np.float32)
    flat.tofile(args.out / "heads.f32")
    (args.out / "heads.json").write_text(json.dumps(description, indent=1))
    print(f"{args.out}: {flat.size:,} parameters; {json.dumps({k: v for k, v in description.items() if k != 'heads'})}")


if __name__ == "__main__":
    main()
