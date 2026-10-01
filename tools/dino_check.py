#!/usr/bin/env python3
"""Play every Dino Arena state through the compiled game model on the DevKit.

The game can only produce the states in `games/dino_policy.py`, so this is an exhaustive test
of the deployed model: every state, the action it picks, and how sure it is.

    python3 tools/dino_check.py [--remote /media/nvme/laya] [--model-dir model-dino]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from games import dino_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="sima@192.168.91.225")
    ap.add_argument("--remote", default="/media/nvme/laya")
    ap.add_argument("--model-dir", default="model-dino")
    args = ap.parse_args()

    rows = dino_policy.all_states()
    requests = "".join(json.dumps({"state": s, "questions": {"action": dino_policy.QUESTION}}) + "\n"
                       for s, _ in rows)
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.board, f"{args.remote}/laya serve {args.remote}/{args.model_dir}"],
        input=requests, capture_output=True, text=True)
    answers = [json.loads(line) for line in result.stdout.splitlines()
               if line.startswith('{"answers"') or line.startswith('{"error"')]
    if result.returncode != 0 or len(answers) != len(rows):
        sys.exit(f"board run failed ({result.returncode}), {len(answers)} answers:\n{result.stdout[-1500:]}")

    wrong, weakest, mla = [], 1.0, []
    for (state, want), got in zip(rows, answers):
        if "error" in got:
            sys.exit("board error: " + got["error"])
        answer = got["answers"]["action"]
        mla.append(got["usage"]["mla_ms"])
        weakest = min(weakest, answer["probabilities"][want])
        if answer["choice"] != want:
            wrong.append(f"  {json.dumps(state)}: wanted {want}, got {answer['choice']} {answer['probabilities']}")
    print(f"{len(rows) - len(wrong)}/{len(rows)} states right; lowest probability on the right action {weakest:.4f}")
    print(f"MLA per decision: mean {sum(mla) / len(mla):.2f} ms  max {max(mla):.2f} ms  "
          f"({got['usage']['tokens']} tokens in a {got['usage']['seq_len']}-token graph)")
    if wrong:
        print("\n".join(wrong))
        sys.exit(1)


if __name__ == "__main__":
    main()
