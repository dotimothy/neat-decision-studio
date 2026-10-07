#!/usr/bin/env python3
"""Compare the chess model on the DevKit with its PyTorch reference.

For every position in the dump of `tools/chess_reference.py`, each legal move's question is
sent through `laya serve`, worded by the same data the browser's wording is checked against.
Reported: how far the board's win chances are from PyTorch's, and in how many positions the
move rated highest is the same one.

    python3 tools/chess_check.py [--reference build/chess_reference.json] [--model-dir model-chess]
"""
import argparse
import json
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="build/chess_reference.json")
    ap.add_argument("--board", default="sima@192.168.91.225")
    ap.add_argument("--remote", default="/media/nvme/laya")
    ap.add_argument("--model-dir", default="model-chess")
    args = ap.parse_args()

    reference = json.load(open(args.reference))
    levels = reference["criteria"]
    asked = [(position, move) for position in reference["positions"] for move in position["moves"]]
    requests = "".join(json.dumps({"state": position["state"], "questions": {"q": {
        "type": "score", "instructions": move["instructions"], "criteria": levels}}}) + "\n" for position, move in asked)
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.board, f"{args.remote}/laya serve {args.remote}/{args.model_dir}"],
        input=requests, capture_output=True, text=True)
    answers = [json.loads(line) for line in result.stdout.splitlines()
               if line.startswith('{"answers"') or line.startswith('{"error"')]
    if result.returncode != 0 or len(answers) != len(asked):
        sys.exit(f"board run failed ({result.returncode}), {len(answers)} answers:\n{result.stdout[-1500:]}")

    errors, mla, tokens_off, by_position = [], [], 0, {}
    for (position, move), got in zip(asked, answers):
        if "error" in got:
            sys.exit("board error: " + got["error"])
        win = (got["answers"]["q"]["score"] + 0.5) / len(levels)         # the levels' midpoints, weighted
        errors.append(abs(win - move["win"]))
        mla.append(got["usage"]["mla_ms"])
        tokens_off += got["usage"]["tokens"] != move["tokens"]
        by_position.setdefault(position["fen"], []).append((move["san"], move["win"], win))
    same = close = 0
    for fen, moves in by_position.items():
        ref_best = max(moves, key=lambda m: m[1])
        board_best = max(moves, key=lambda m: m[2])
        same += ref_best[0] == board_best[0]
        close += ref_best[1] - board_best[1] <= 0.01            # the board's pick is within a point of the best
        if ref_best[0] != board_best[0]:
            print(f"  {fen}: PyTorch {ref_best[0]} {ref_best[1]:.3f}, board {board_best[0]} (PyTorch rates it {board_best[1]:.3f})")
    errors.sort()
    n = len(errors)
    print(f"{n} questions in {len(by_position)} positions, on the {answers[0]['usage']['seq_len']}-token graph")
    print(f"win chance, board against PyTorch: mean {sum(errors) / n:.4f}, p95 {errors[int(n * 0.95)]:.4f}, max {errors[-1]:.4f} (of 1)")
    print(f"same best move in {same}/{len(by_position)} positions; the board's choice within one point of PyTorch's best in {close}")
    print(f"token counts that differ from upstream's: {tokens_off}")
    print(f"MLA per decision: mean {sum(mla) / n:.2f} ms; a move of 30 legal moves takes about {30 * sum(mla) / n / 1000:.1f} s")
    sys.exit(1 if tokens_off else 0)


if __name__ == "__main__":
    main()
