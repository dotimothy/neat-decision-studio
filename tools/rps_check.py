#!/usr/bin/env python3
"""Ask a model on the DevKit what a Rock Paper Scissors opponent throws next.

300 histories of players with a favourite throw. There is no right answer to check, so this
reports how the model reads a history: how often it names one of the most used throws, how
often the latest one, and what playing the throw that beats its answer would have won against
the throw that player makes next.

    python3 tools/rps_check.py [--remote /media/nvme/laya] [--model-dir model]
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from games import rps_policy as rps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="sima@192.168.91.225")
    ap.add_argument("--remote", default="/media/nvme/laya")
    ap.add_argument("--model-dir", default="model")
    args = ap.parse_args()

    histories = rps.histories()
    requests = "".join(json.dumps({"state": rps.state(h[:-1]), "questions": {"q": rps.question()}}) + "\n" for h in histories)
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.board,
         f"{args.remote}/laya serve {args.remote}/{args.model_dir} --seq-lens 128"],
        input=requests, capture_output=True, text=True)
    answers = [json.loads(line) for line in result.stdout.splitlines()
               if line.startswith('{"answers"') or line.startswith('{"error"')]
    if result.returncode != 0 or len(answers) != len(histories):
        sys.exit(f"board run failed ({result.returncode}), {len(answers)} answers:\n{result.stdout[-1500:]}")

    most = latest = right = won = lost = 0
    for history, got in zip(histories, answers):
        if "error" in got:
            sys.exit("board error: " + got["error"])
        seen, coming = history[:-1][-rps.MEMORY:], history[-1]      # what it was shown, and the throw that followed
        guess = got["answers"]["q"]["choice"]
        counts = {throw: seen.count(throw) for throw in rps.THROWS}
        most += counts[guess] == max(counts.values())
        latest += guess == seen[-1]
        right += guess == coming
        mine = next(throw for throw in rps.THROWS if rps.BEATS[throw] == guess)
        won += rps.BEATS[mine] == coming
        lost += rps.BEATS[coming] == mine
    n = len(histories)
    print(f"names one of the opponent's most used throws: {most}/{n}")
    print(f"names the opponent's latest throw: {latest}/{n}")
    print(f"names the throw that came next: {right}/{n} (a third by chance)")
    print(f"playing what beats its answer: won {won}, tied {n - won - lost}, lost {lost} of {n}")
    print(f"MLA per decision: mean {sum(a['usage']['mla_ms'] for a in answers) / n:.2f} ms ({answers[0]['usage']['tokens']} tokens)")


if __name__ == "__main__":
    main()
