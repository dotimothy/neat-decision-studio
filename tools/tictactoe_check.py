#!/usr/bin/env python3
"""Run every Tic-Tac-Toe question the game can ask through a model on the DevKit.

Every reachable position is described both ways the harness can look (one move ahead, and to
the end of the game); each distinct question is asked once. The model should take a square
with the best description on offer.

    python3 tools/tictactoe_check.py [--remote /media/nvme/laya] [--model-dir model]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from games import tictactoe_policy as ttt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="sima@192.168.91.225")
    ap.add_argument("--remote", default="/media/nvme/laya")
    ap.add_argument("--model-dir", default="model")
    args = ap.parse_args()

    asked = {}                        # (look, the question's options) -> levels
    for board, mark in ttt.all_positions():
        for deep in (False, True):
            levels = ttt.outcomes(board, mark, deep)
            if len(set(levels.values())) > 1:                # otherwise any square is as good
                asked[("to the end" if deep else "one move ahead", tuple(levels.items()))] = levels
    requests = "".join(json.dumps({"state": ttt.STATE, "questions": {"q": ttt.question(levels)}}) + "\n" for levels in asked.values())
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.board,
         f"{args.remote}/laya serve {args.remote}/{args.model_dir} --seq-lens 128"],
        input=requests, capture_output=True, text=True)
    answers = [json.loads(line) for line in result.stdout.splitlines()
               if line.startswith('{"answers"') or line.startswith('{"error"')]
    if result.returncode != 0 or len(answers) != len(asked):
        sys.exit(f"board run failed ({result.returncode}), {len(answers)} answers:\n{result.stdout[-1500:]}")

    tally, misses, mla, tokens = {}, {}, [], 0
    for (look, _), levels, got in zip(asked, asked.values(), answers):
        if "error" in got:
            sys.exit("board error: " + got["error"])
        picked, best = levels[got["answers"]["q"]["choice"]], min(levels.values())
        mla.append(got["usage"]["mla_ms"])
        tokens = max(tokens, got["usage"]["tokens"])
        count = tally.setdefault((look, ttt.OUTCOMES[best]), [0, 0])
        count[0] += picked == best
        count[1] += 1
        if picked != best:
            key = (look, f"{ttt.OUTCOMES[picked]!r} instead of {ttt.OUTCOMES[best]!r}")
            misses[key] = misses.get(key, 0) + 1
    for look in ("one move ahead", "to the end"):
        right = sum(count[0] for (l, _), count in tally.items() if l == look)
        total = sum(count[1] for (l, _), count in tally.items() if l == look)
        print(f"looking {look}: a best-described square taken in {right}/{total} questions")
        for (l, best), (ok, n) in tally.items():
            if l == look:
                print(f"    when the best on offer is {best!r}: {ok}/{n}")
        for (l, what), n in sorted(misses.items(), key=lambda item: -item[1]):
            if l == look:
                print(f"    took {what}: {n}")
    print(f"MLA per decision: mean {sum(mla) / len(mla):.2f} ms; longest question {tokens} tokens")


if __name__ == "__main__":
    main()
