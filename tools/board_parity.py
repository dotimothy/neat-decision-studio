#!/usr/bin/env python3
"""Compare the compiled ELF on the DevKit with the PyTorch golden dump.

Feeds the reference token ids straight to `laya raw`, so a difference is the MLA graph (or the
runtime's buffer handling), never the tokenizer. Tokenization is checked separately by
`--text`, which runs the full `laya run` path and compares answers.

    python3 tools/board_parity.py                 # raw logits
    python3 tools/board_parity.py --text          # end-to-end answers

Needs only numpy and ssh access to the board.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def ssh(board: str, command: str, stdin: str | None = None) -> str:
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", board, command], input=stdin,
                            capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"board command failed ({result.returncode}): {command}\n{result.stderr[-2000:]}")
    return result.stdout


def json_tail(text: str):
    """The runtime logs to stdout before its JSON document; take the document."""
    return json.loads(text[text.index("\n{") + 1:] if not text.startswith("{") else text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, default=Path("build/reference.npz"))
    ap.add_argument("--board", default="sima@192.168.91.225")
    ap.add_argument("--remote", default="/media/nvme/laya")
    ap.add_argument("--model-dir", default="model", help="model directory under --remote")
    ap.add_argument("--seq-len", type=int, default=0, help="pin one compiled graph")
    ap.add_argument("--text", action="store_true", help="check the full text path instead")
    ap.add_argument("--tolerance", type=float, default=0.35,
                    help="largest accepted |logit difference|, for logits up to 3 in size and "
                         "proportionally more above that (bfloat16 through the whole network)")
    args = ap.parse_args()
    ref = np.load(args.reference)
    meta = json.loads(args.reference.with_suffix(".json").read_text())
    laya, model = f"{args.remote}/laya", f"{args.remote}/{args.model_dir}"

    if args.text:
        failed = False
        for case in meta:
            ssh(args.board, f"cat > {args.remote}/parity_state.txt", case["state"])
            ssh(args.board, f"cat > {args.remote}/parity_question.json",
                json.dumps({"q": case["question"]}))
            out = json_tail(ssh(args.board, f"{laya} run {model} --state-file {args.remote}/parity_state.txt "
                                            f"--questions {args.remote}/parity_question.json"))
            got, want = out["answers"]["q"], case["answer"]
            if out["usage"]["tokens"] != case["tokens"]:
                failed = True
                print(f"{case['name']}: TOKEN COUNT {out['usage']['tokens']} != {case['tokens']}")
            key = got["type"]
            same = got[key] == want[key] if key == "choice" else abs(got[key] - want[key]) < 0.1
            failed |= not same
            print(f"{case['name']}: {key}={got[key]} (torch {want[key]}) "
                  f"answer_confidence {got['answer_confidence']} (torch {want['answer_confidence']}) "
                  f"seq_len {out['usage']['seq_len']} mla {out['usage']['mla_ms']} ms "
                  f"{'ok' if same else 'MISMATCH'}")
        sys.exit(1 if failed else 0)

    names = sorted({k.split(".")[0] for k in ref.files})
    cases = [{"name": n, "ids": ref[n + ".ids"].tolist(), "markers": ref[n + ".markers"].tolist(),
              "qtype": int(ref[n + ".qtype"])} for n in names]
    ssh(args.board, f"cat > {args.remote}/parity_raw.json", json.dumps({"cases": cases}))
    pin = f" --seq-len {args.seq_len}" if args.seq_len else ""
    out = json_tail(ssh(args.board, f"{laya} raw {model} --input {args.remote}/parity_raw.json{pin}"))
    worst, checked = 0.0, 0
    for result in out["results"]:
        name = result["name"]
        if "skipped" in result:
            print(f"{name}: {result['tokens']} tokens, skipped ({result['skipped']})")
            continue
        logits, want = np.array(result["logits"]), ref[name + ".logits"]
        err = float(np.abs(logits - want).max())
        agree = int(logits.argmax()) == int(want.argmax())
        # bfloat16 error scales with the logit: judge it relative to the largest one.
        scaled = err / max(1.0, float(np.abs(want).max()) / 3.0)
        worst, checked = max(worst, scaled if agree else np.inf), checked + 1
        print(f"{name}: tokens {result['tokens']} seq_len {result['seq_len']} mla {result['mla_ms']:.1f} ms\n"
              f"    mla   {logits.round(3)}  act {np.array(result['act_logits']).round(1)}\n"
              f"    torch {want.round(3)}  act {ref[name + '.act_logits'].round(1)}\n"
              f"    max|d| {err:.3f}  argmax {'same' if agree else 'DIFFERENT'}")
    ok = checked > 0 and worst < args.tolerance
    print("PASS" if ok else "FAIL", f"worst {worst:.3f} over {checked} cases")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
