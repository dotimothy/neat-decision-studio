"""Pac-Man: what the model is asked, and why it is asked that way.

The game and its geometry are in `webapp/static/pacman-core.js`; this module holds the same
texts for `tools/pacman_check.py`, which runs every combination through the board. The two
must stay the same.

Pac-Man uses a general Laya checkpoint as it ships, the way Snake does. The model is not
shown the maze. The options of a `choice` question are the ways Pac-Man can go from where it
stands (two to four of up, left, down and right), and the harness describes each with one of
four phrases: whether a ghost catches Pac-Man on the next step (it tries the step), whether
it closes in on a ghost that is near, and otherwise whether the shortest way to the food gets
shorter. The food is a blue ghost that can still be caught, if there is one, and otherwise
the dots. Picking the best-described way is the model's part. The options are named "go up",
"go left", "go down" and "go right": named plainly, "right" was read as "correct", and offered
"down" toward the food and "right" away from it the model took "right" every time.

The phrases are Snake's, with a ghost where Snake has a wall. A first wording, with five
phrases (a blue ghost apart from the dots, "dots" for food, "toward a ghost" and "into a
ghost" for the dangers), had the general model pick a best-described way in 802 of 1203
positions and a dangerous one 30 times; with these, and the options named as below, it
picks one in all 536 and a dangerous one never.
"""
from itertools import combinations, product

MOVES = ["up", "left", "down", "right"]
OPTIONS = {move: f"go {move}" for move in MOVES}          # the options' names in the question
STATE = "Pac-Man needs a safe move that brings it closer to the food."
INSTRUCTIONS = "Which way should Pac-Man go?"
# What a move can lead to, best first.
OUTCOMES = ["safe and toward the food", "safe but away from the food", "a ghost is near, dangerous",
            "caught by a ghost, Pac-Man dies"]
RISKY_FROM = 2          # outcomes from this index on put Pac-Man in danger


def question(outcomes: dict[str, int]) -> dict:
    """`outcomes`: the open ways, each with an index into OUTCOMES, in the order of MOVES."""
    return {"type": "choice", "instructions": INSTRUCTIONS,
            "criteria": {OPTIONS[move]: OUTCOMES[level] for move, level in outcomes.items()}}


def all_positions() -> list[dict[str, int]]:
    """Every combination of open ways (two to four) and of what each leads to."""
    return [dict(zip(ways, levels))
            for count in range(2, len(MOVES) + 1) for ways in combinations(MOVES, count)
            for levels in product(range(len(OUTCOMES)), repeat=count)]
