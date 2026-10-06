"""Sudoku: what the model is asked, and why it is asked that way.

The puzzle and its logic are in `webapp/static/sudoku-core.js`; this module holds the same
texts for `tools/sudoku_check.py`, which runs them through the board. The two must stay the
same.

Sudoku uses a general Laya checkpoint as it ships. The obvious way to ask does not work: given
the digits a cell's row, column and box already hold ("Row: 2, 5. Column: 1, 8, 9. Box: 3, 4,
7.") and asked which digit is missing, the model answered with a digit from the lists in 399 of
400 tries. It matches options against the text, so it finds what is there and not what is
absent; the digit it finds least is the missing one only about two times in three.

So, as in Snake, the harness does the looking and the options of a `choice` question say what
it found. Filling a cell takes two decisions:

  which cell    five empty cells, each described by how sure its digit is
  which digit   the nine digits, each described by what it would do in that cell

Preferring "safe and certain" over a risk, and a risk over a digit that breaks the puzzle, is
the model's part. The wording was chosen on the board among several: the best option has to
echo the state ("needs ... safe and certain"), as it does in Snake. With "fits" against "taken
by the row", or "correct" against "wrong", the model picked a rule-breaking digit whenever no
digit was certain.

What the model does not manage is to rank the two kinds of risk: with no certain cell on
offer it takes "a big risk" over "a small risk" in a share of cases (see the check). That only
matters in hard puzzles, where it costs some mistakes.
"""
import random
from itertools import product

CELL_STATE = "The puzzle needs a cell that is safe and certain."
CELL_INSTRUCTIONS = "Which cell should be filled next?"
# How sure a cell's digit is, best first.
CELL_OUTCOMES = ["safe and certain, one digit must go here", "a small risk, two digits fit", "a big risk, many digits fit"]
DIGIT_STATE = "The cell needs a digit that is safe and certain."
DIGIT_INSTRUCTIONS = "Which digit should go in the cell?"
# What a digit would do in the cell, best first.
DIGIT_OUTCOMES = ["safe and certain", "safe but a guess", "repeats in the row, breaks the puzzle",
                  "repeats in the column, breaks the puzzle", "repeats in the box, breaks the puzzle"]
BREAKS_FROM = 2          # digit outcomes from this index on break the rules
OFFERED = 5
DIGITS = [str(digit) for digit in range(1, 10)]
CELLS = [f"r{row}c{column}" for row in range(1, 10) for column in range(1, 10)]


def cell_question(cells: list[str], outcomes: tuple[int, ...]) -> dict:
    return {"type": "choice", "instructions": CELL_INSTRUCTIONS,
            "criteria": {cell: CELL_OUTCOMES[level] for cell, level in zip(cells, outcomes)}}


def digit_question(outcomes: list[int]) -> dict:
    return {"type": "choice", "instructions": DIGIT_INSTRUCTIONS,
            "criteria": {digit: DIGIT_OUTCOMES[level] for digit, level in zip(DIGITS, outcomes)}}


def cell_positions(seed: int = 5) -> list[tuple[list[str], tuple[int, ...]]]:
    """Every combination of outcomes for two to five offered cells, on cells drawn at random."""
    rng = random.Random(seed)
    return [(rng.sample(CELLS, count), outcomes)
            for count in range(2, OFFERED + 1) for outcomes in product(range(len(CELL_OUTCOMES)), repeat=count)]


def digit_positions(count: int = 300, seed: int = 5) -> list[tuple[str, list[int]]]:
    """Cells of the three kinds the game produces, with the digits ruled out at random.

    only    one digit is left for the cell
    place   the cell is the only place for a digit, and one to three other digits fit as well
    risk    two to four digits fit and none is certain
    """
    rng = random.Random(seed)
    positions = []
    for index in range(count):
        kind = ("only", "place", "risk")[index % 3]
        outcomes = [rng.choice([2, 3, 4]) for _ in DIGITS]
        spots = rng.sample(range(9), 4)
        if kind == "only":
            outcomes[spots[0]] = 0
        elif kind == "place":
            outcomes[spots[0]] = 0
            for spot in spots[1:rng.randint(2, 4)]:
                outcomes[spot] = 1
        else:
            for spot in spots[:rng.randint(2, 4)]:
                outcomes[spot] = 1
        positions.append((kind, outcomes))
    # The longest question there can be: it has to fit the 128-token graph.
    positions.append(("only", [3, 3, 3, 3, 0, 3, 3, 3, 3]))
    return positions
