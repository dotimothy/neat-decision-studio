#!/usr/bin/env python3
"""Run the Sudoku questions through a model on the DevKit.

A cell is filled in two decisions. For "which cell", every combination of descriptions for two
to five offered cells is tried; for "which digit", 300 cells of the three kinds the game
produces. Each time the model should pick an option with the best description on offer. A
miss matters most when a certain cell or digit was offered and it took something else.

    python3 tools/sudoku_check.py [--remote /media/nvme/laya] [--model-dir model]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from games import sudoku_policy as sudoku


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="sima@192.168.91.225")
    ap.add_argument("--remote", default="/media/nvme/laya")
    ap.add_argument("--model-dir", default="model")
    args = ap.parse_args()

    asked = []                      # (question kind, case kind, {option: outcome}, request)
    for cells, outcomes in sudoku.cell_positions():
        asked.append(("cell", "certain on offer" if 0 in outcomes else "risks only", dict(zip(cells, outcomes)),
                      {"state": sudoku.CELL_STATE, "questions": {"q": sudoku.cell_question(cells, outcomes)}}))
    for kind, outcomes in sudoku.digit_positions():
        asked.append(("digit", kind, dict(zip(sudoku.DIGITS, outcomes)),
                      {"state": sudoku.DIGIT_STATE, "questions": {"q": sudoku.digit_question(outcomes)}}))
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.board,
         f"{args.remote}/laya serve {args.remote}/{args.model_dir} --seq-lens 128"],
        input="".join(json.dumps(request) + "\n" for *_, request in asked), capture_output=True, text=True)
    answers = [json.loads(line) for line in result.stdout.splitlines()
               if line.startswith('{"answers"') or line.startswith('{"error"')]
    if result.returncode != 0 or len(answers) != len(asked):
        sys.exit(f"board run failed ({result.returncode}), {len(answers)} answers:\n{result.stdout[-1500:]}")

    tally, mla, tokens, broke, failed = {}, [], {}, 0, False
    for (question, kind, outcomes, _), got in zip(asked, answers):
        if "error" in got:
            sys.exit("board error: " + got["error"])
        picked = outcomes[got["answers"]["q"]["choice"]]
        mla.append(got["usage"]["mla_ms"])
        tokens[question] = max(tokens.get(question, 0), got["usage"]["tokens"])
        count = tally.setdefault((question, kind), [0, 0])
        count[0] += picked == min(outcomes.values())
        count[1] += 1
        broke += question == "digit" and picked >= sudoku.BREAKS_FROM
        # With a certain option on offer there is one right answer; among risks there is none.
        failed |= picked != min(outcomes.values()) and min(outcomes.values()) == 0
    for (question, kind), (best, total) in tally.items():
        print(f"which {question}, {kind}: a best-described option picked {best}/{total}")
    print(f"a digit that breaks the puzzle picked: {broke}")
    print(f"MLA per decision: mean {sum(mla) / len(mla):.2f} ms; longest question {tokens['cell']} tokens (cell), "
          f"{tokens['digit']} tokens (digit)")
    sys.exit(1 if failed or broke else 0)


if __name__ == "__main__":
    main()
