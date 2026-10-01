#!/usr/bin/env python3
"""Check a generated Laya ONNX graph against the PyTorch golden dump.

Runs under the model-compiler venv (onnxruntime, no torch):

    python tools/verify_onnx.py --build build/laya --seq_len 128
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from laya_sima import hostio
from laya_sima.config import LayaConfig
from laya_sima.weights import LayaWeights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("models/laya"))
    ap.add_argument("--build", type=Path, default=Path("build/laya"))
    ap.add_argument("--reference", type=Path, default=Path("build/reference.npz"))
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--tolerance", type=float, default=2e-2)
    args = ap.parse_args()

    cfg = LayaConfig.from_checkpoint(args.model)
    weights = LayaWeights(args.model)
    table = weights.raw("encoder.embeddings.tok_embeddings.weight")
    act0 = weights.raw("act_head.0.weight")
    act_tail = {"w_feats": act0[:, cfg.hidden_size:], "w_out": weights.raw("act_head.2.weight"),
                "b_out": weights.raw("act_head.2.bias")}
    ref = np.load(args.reference)
    sess = ort.InferenceSession(str(args.build / "onnx_files" / f"laya_s{args.seq_len}.onnx"))

    worst = 0.0
    for name in sorted({k.split(".")[0] for k in ref.files}):
        ids, markers = ref[name + ".ids"], ref[name + ".markers"]
        if len(ids) > args.seq_len:
            print(f"{name}: {len(ids)} tokens, skipped at seq_len {args.seq_len}")
            continue
        global_mask, local_mask = hostio.attention_masks(len(ids), args.seq_len, cfg.sliding_window)
        scores, act_pre = sess.run(None, {
            "embeds": hostio.embed(ids, table, args.seq_len, cfg.pad_token_id),
            "global_mask": global_mask, "local_mask": local_mask,
            "qtype": hostio.qtype_one_hot(int(ref[name + ".qtype"]), args.seq_len, cfg.qtype_channels),
        })
        logits, act = hostio.decode(scores, act_pre, markers, act_tail)
        err = np.abs(logits - ref[name + ".logits"]).max()
        act_err = (np.abs(act - ref[name + ".act_logits"]) / np.abs(ref[name + ".act_logits"])).max()
        worst = max(worst, err, act_err)
        print(f"{name}: tokens {len(ids)} logits {logits.round(3)} ref {ref[name + '.logits'].round(3)} "
              f"max|d| {err:.2e}  act rel {act_err:.2e}")
    print("PASS" if worst < args.tolerance else "FAIL", f"worst {worst:.2e}")
    sys.exit(0 if worst < args.tolerance else 1)


if __name__ == "__main__":
    main()
