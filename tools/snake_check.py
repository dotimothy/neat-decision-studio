#!/usr/bin/env python3
"""Run every Snake question the game can ask through a model on the DevKit.

A position is described by what each of the three moves leads to, so there are 64
combinations. For each the model should pick a move with the best description on offer. A
miss matters most when it picks a fatal move while a safe one exists.

    python3 tools/snake_check.py [--remote /media/nvme/laya] [--model-dir model]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from games import snake_policy as snake


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="sima@192.168.91.225")
    ap.add_argument("--remote", default="/media/nvme/laya")
    ap.add_argument("--model-dir", default="model")
    args = ap.parse_args()

    positions = snake.all_positions()
    requests = "".join(json.dumps({"state": snake.STATE, "questions": {"move": snake.question(p)}}) + "\n"
                       for p in positions)
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.board,
         f"{args.remote}/laya serve {args.remote}/{args.model_dir} --seq-lens 128"],
        input=requests, capture_output=True, text=True)
    answers = [json.loads(line) for line in result.stdout.splitlines()
               if line.startswith('{"answers"') or line.startswith('{"error"')]
    if result.returncode != 0 or len(answers) != len(positions):
        sys.exit(f"board run failed ({result.returncode}), {len(answers)} answers:\n{result.stdout[-1500:]}")

    best = fatal = decidable = 0
    misses, mla = [], []
    for position, got in zip(positions, answers):
        if "error" in got:
            sys.exit("board error: " + got["error"])
        answer = got["answers"]["move"]
        mla.append(got["usage"]["mla_ms"])
        picked = position[snake.MOVES.index(answer["choice"])]
        if min(position) >= snake.FATAL_FROM:
            continue                                   # no safe move: nothing to get right
        decidable += 1
        best += picked == min(position)
        if picked >= snake.FATAL_FROM:
            fatal += 1
        if picked != min(position):
            options = " / ".join(snake.OUTCOMES[level] for level in position)
            misses.append(f"  [{options}] -> {answer['choice']} ({snake.OUTCOMES[picked]})")
    print(f"{best}/{decidable} positions with a safe move: a best-described move picked")
    print(f"{fatal}/{decidable} of them: a fatal move picked although a safe one was offered")
    print(f"MLA per decision: mean {sum(mla) / len(mla):.2f} ms ({got['usage']['tokens']} tokens)")
    if misses:
        print("not the best move:")
        print("\n".join(misses))
    sys.exit(1 if fatal else 0)


if __name__ == "__main__":
    main()
