#!/usr/bin/env python3
"""Run every Pac-Man question the game can ask through a model loaded in the app on the DevKit.

A position is described by its open ways (two to four) and what each leads to, one of four
descriptions, so there are 608 combinations. For each the model should pick a way with the
best description on offer. A miss matters most when it picks a dangerous or fatal way while a
safe one is offered. The questions go to the running app (`/api/predict`), so nothing more is
loaded on the MLA than the model already there.

    python3 tools/pacman_check.py [--model general] [--url http://<board-ip>:8095]
"""
import argparse
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from games import pacman_policy as pacman


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="general", help="a model loaded in the app")
    ap.add_argument("--url", default="http://192.168.91.225:8095", help="the app on the board")
    args = ap.parse_args()

    best = risky = decidable = 0
    misses, mla, by_count = Counter(), [], Counter()
    for position in pacman.all_positions():
        request = {"model": args.model, "state": pacman.STATE, "questions": {"move": pacman.question(position)}}
        with urllib.request.urlopen(urllib.request.Request(f"{args.url}/api/predict", json.dumps(request).encode(),
                                                           {"Content-Type": "application/json"}), timeout=60) as response:
            got = json.load(response)
        if "error" in got:
            sys.exit("board error: " + got["error"])
        mla.append(got["usage"]["mla_ms"])
        picked = position[next(move for move, name in pacman.OPTIONS.items() if name == got["answers"]["move"]["choice"])]
        top = min(position.values())
        if top >= pacman.RISKY_FROM:
            continue                                   # no safe way: nothing to get right
        decidable += 1
        by_count[len(position)] += picked == top
        best += picked == top
        risky += picked >= pacman.RISKY_FROM
        if picked != top:
            misses[(pacman.OUTCOMES[top], pacman.OUTCOMES[picked])] += 1
    print(f"{best}/{decidable} positions with a safe way: a best-described way picked "
          f"({', '.join(f'{n} ways: {by_count[n]}' for n in sorted(by_count))})")
    print(f"{risky}/{decidable} of them: a dangerous or fatal way picked although a safe one was offered")
    print(f"MLA per decision: mean {sum(mla) / len(mla):.2f} ms")
    if misses:
        print("not the best way (best on offer -> picked: times):")
        for (top, picked), n in misses.most_common():
            print(f"  {top} -> {picked}: {n}")
    sys.exit(1 if risky else 0)


if __name__ == "__main__":
    main()
