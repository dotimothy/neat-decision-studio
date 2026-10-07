#!/usr/bin/env python3
"""Compare CLM on the DevKit with the PyTorch reference (tools/clm_reference.py).

Two comparisons, both from the reference's token ids and questions:

    hidden    what the compiled encoder embeds each reference text to, against transformers:
              the cosine between the two (the heads only see the direction)
    answers   the reference cases asked of `laya serve`, against the reference answers: the
              choice made, and the largest difference in probability

    python3 tools/clm_check.py                      # both
    python3 tools/clm_check.py --graphs 1           # a directory holding only the first graph:
                                                    # its output against the reference after as
                                                    # many layers (build/clm_reference/layers.npz)

Needs only numpy and ssh access to the board.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def ssh(board: str, command: str, stdin: str | None = None) -> str:
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", board, command], input=stdin, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"board command failed ({result.returncode}): {command}\n{(result.stdout + result.stderr)[-2000:]}")
    return result.stdout


def last_json(text: str):
    return json.loads([line for line in text.splitlines() if line.startswith("{")][-1])


def board_hidden(args, id_lists: list[list[int]]) -> tuple[np.ndarray, float]:
    laya, model = f"{args.remote}/laya", f"{args.remote}/{args.model_dir}"
    ssh(args.board, f"cat > {args.remote}/clm_check_ids.json", json.dumps({"cases": [{"ids": ids} for ids in id_lists]}))
    out = last_json(ssh(args.board, f"{laya} hidden {model} --input {args.remote}/clm_check_ids.json --out {args.remote}/clm_check_hidden.f32"))
    raw = subprocess.run(["ssh", "-o", "BatchMode=yes", args.board, f"cat {args.remote}/clm_check_hidden.f32"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(len(id_lists), -1), out["mla_ms_mean"]


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, default=Path("build/clm_reference"))
    ap.add_argument("--board", default="sima@192.168.91.225")
    ap.add_argument("--remote", default="/media/nvme/laya")
    ap.add_argument("--model-dir", default="model-clm", help="model directory under --remote")
    ap.add_argument("--graphs", type=int, help="the directory holds only this many of the graphs")
    ap.add_argument("--layers-per-graph", type=int, default=4)
    args = ap.parse_args()

    if args.graphs:
        cases = json.loads((args.reference / "layer_cases.json").read_text())["cases"]
        layers = np.load(args.reference / "layers.npz")
        got, mla_ms = board_hidden(args, [case["ids"] for case in cases])
        depth = args.graphs * args.layers_per_graph
        print(f"after {depth} layers ({args.graphs} graph{'s' * (args.graphs > 1)}, {mla_ms:.1f} ms a text):")
        for case, row in zip(cases, got):
            want = layers[str(case["text"])][depth]
            print(f"  {len(case['ids']):3d} tokens  cosine {cosine(row, want):.5f}  "
                  f"max |difference| {np.abs(row - want).max():.3f} of {np.abs(want).max():.1f}")
        return

    reference = json.loads((args.reference / "cases.json").read_text())
    want = np.load(args.reference / "hidden.npy")
    got, mla_ms = board_hidden(args, [text["ids"] for text in reference["texts"]])
    cosines = cosine(got, want)
    print(f"hidden: {len(cosines)} texts, {mla_ms:.1f} ms a text on the MLA; cosine with transformers "
          f"mean {cosines.mean():.5f}, worst {cosines.min():.5f}")

    requests = "".join(json.dumps({"state": case["state"], "questions": case["questions"]}) + "\n"
                       for case in reference["cases"])
    out = ssh(args.board, f"{args.remote}/laya serve {args.remote}/{args.model_dir}", requests)
    replies = [json.loads(line) for line in out.splitlines() if line.startswith('{"answers"')]
    same = total = 0
    worst = 0.0
    for case, reply in zip(reference["cases"], replies, strict=True):
        print(f"{case['name']}  ({reply['usage']['latency_ms']:.0f} ms, {reply['usage']['encoder_passes']} texts embedded)")
        for qid, ref in case["answers"].items():
            ans = reply["answers"][qid]
            kind = ref["type"]
            if kind == "noul":
                ref_label, label, gap = ref["noul"] >= 0.5, ans["noul"] >= 0.5, abs(ref["noul"] - ans["noul"])
                shown = f"{ans['noul']:.3f} (reference {ref['noul']:.3f})"
            else:
                top = lambda p: max(p, key=p.get)
                ref_label, label = top(ref["probabilities"]), top(ans["probabilities"])
                gap = max(abs(ref["probabilities"][k] - ans["probabilities"][k]) for k in ref["probabilities"])
                shown = f"{label} {ans['probabilities'][label]:.3f} (reference {ref_label} {ref['probabilities'][ref_label]:.3f})"
            total += 1
            same += label == ref_label
            worst = max(worst, gap)
            print(f"    {'ok ' if label == ref_label else 'DIFFERS'} {qid}: {shown}")
    print(f"answers: {same}/{total} the same as the reference; largest difference in a probability {worst:.3f}")


if __name__ == "__main__":
    main()
