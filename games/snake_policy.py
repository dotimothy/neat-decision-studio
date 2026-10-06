"""Snake: what the model is asked, and why it is asked that way.

The game and its geometry are in `webapp/static/snake-core.js`; this module holds the same
texts for `tools/snake_check.py`, which runs every combination through the board. The two
must stay the same.

Snake uses a general Laya checkpoint as it ships. The model is not shown the board: it could
not read one. Instead the options of a `choice` question are built per position. For each of
the three moves (left, straight, right, relative to the snake's heading) the harness works
out where the move leads and describes it with one of four phrases, and the model picks an
option. Laya assembles its answer space per request, so options that change every move cost
nothing.

What the harness computes is geometry: whether the square is free, whether the shortest path
to the food gets shorter, and whether the space beyond the square is smaller than the snake.
Preferring "safe and toward the food" over "safe but away" over the fatal ones is the model's
part.
"""
from itertools import product

MOVES = ["left", "straight", "right"]
STATE = "The snake needs a safe move that brings it closer to the food."
INSTRUCTIONS = "Which way should the snake go?"
# What a move can lead to, best first.
OUTCOMES = ["safe and toward the food", "safe but away from the food",
            "a dead end, the snake dies", "blocked, the snake dies"]
FATAL_FROM = 2          # outcomes from this index on lose the game


def question(outcomes: tuple[int, int, int]) -> dict:
    return {"type": "choice", "instructions": INSTRUCTIONS,
            "criteria": {move: OUTCOMES[level] for move, level in zip(MOVES, outcomes)}}


def all_positions() -> list[tuple[int, int, int]]:
    """Every combination of outcomes for the three moves."""
    return list(product(range(len(OUTCOMES)), repeat=len(MOVES)))
