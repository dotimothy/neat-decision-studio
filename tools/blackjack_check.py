#!/usr/bin/env python3
"""Run every blackjack question the game can ask through a model on the DevKit.

There are 15 distinct texts (8 hit-or-stand descriptions, 3 doubling, 4 bet). For each it
reports the rule the model picks against the rule the description is meant to select, then
prints the hit-or-stand chart the model plays from a fresh deck against basic strategy. The
hands the three-fact description cannot tell apart (hard 12 against a 2 or 3, soft 18 against
a 9, 10 or ace) are expected differences.

    python3 tools/blackjack_check.py [--remote /media/nvme/laya] [--model-dir model]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from games import blackjack_policy as bj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="sima@192.168.91.225")
    ap.add_argument("--remote", default="/media/nvme/laya")
    ap.add_argument("--model-dir", default="model")
    args = ap.parse_args()

    cases = bj.descriptions()
    requests = "".join(json.dumps({"state": text, "questions": {"rule": question}}) + "\n"
                       for text, question, _, _ in cases)
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.board,
         f"{args.remote}/laya serve {args.remote}/{args.model_dir} --seq-lens 128"],
        input=requests, capture_output=True, text=True)
    answers = [json.loads(line) for line in result.stdout.splitlines()
               if line.startswith('{"answers"') or line.startswith('{"error"')]
    if result.returncode != 0 or len(answers) != len(cases):
        sys.exit(f"board run failed ({result.returncode}), {len(answers)} answers:\n{result.stdout[-1500:]}")

    played, right = {}, 0
    for (text, _, actions, intended), got in zip(cases, answers):
        if "error" in got:
            sys.exit("board error: " + got["error"])
        answer = got["answers"]["rule"]
        ok = actions[answer["choice"]] == actions[intended]
        right += ok
        played[text] = actions[answer["choice"]]
        print(f"  {'ok  ' if ok else 'MISS'} {text:74s} -> {answer['choice']:15s} ({answer['answer_confidence']:.2f}) "
              f"= {played[text]}  [{got['usage']['mla_ms']:.1f} ms]")
    print(f"\n{right}/{len(cases)} descriptions answered with the intended action")

    hands = bj.all_hands()
    agree = sum(played[bj.play_facts(*hand)] == bj.basic_strategy(*hand) for hand in hands)
    print(f"{agree}/{len(hands)} hit-or-stand hands played as basic strategy would ({100 * agree / len(hands):.1f}%)")
    header = "  ".join("A" if up == bj.ACE else str(up) for up in bj.UPCARDS)
    for soft, totals in ((False, range(5, 22)), (True, range(13, 22))):
        print(f"\n{'soft' if soft else 'hard'}     dealer: {header}      (H hit, s stand, * differs from basic strategy)")
        for total in totals:
            cells = []
            for up in bj.UPCARDS:
                action = played[bj.play_facts(total, soft, up)]
                mark = "*" if action != bj.basic_strategy(total, soft, up) else " "
                cells.append(("H" if action == "hit" else "s") + mark)
            print(f"  {total:2d}             {' '.join(cells)}")
    sys.exit(0 if right == len(cases) else 1)


if __name__ == "__main__":
    main()
