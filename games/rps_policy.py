"""Rock Paper Scissors: what the model is asked, and why it is asked that way.

The game is in `webapp/static/rps-core.js`; this module holds the same texts for
`tools/rps_check.py`. The two must stay the same.

Rock Paper Scissors uses a general Laya checkpoint as it ships, and it is the one game here in
which the model reads the raw material itself. It is given the opponent's last throws as plain
words ("rock, paper, rock, rock") and asked which throw comes next. A Laya model answers with
what it finds in the text, which made it useless for finding the digit missing from a Sudoku
cell and makes it useful here: asked this, the English model names one of the opponent's most
used throws about four times in five (chance is two in five). The harness then plays the throw
that beats the model's answer. That last step is arithmetic and is not asked of the model.

Against a player with a favourite throw this wins more than it loses; against throws made at
random nothing can, and it does not.
"""
import random

THROWS = ["rock", "paper", "scissors"]
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}      # a throw, and what it beats
MEMORY = 8
INSTRUCTIONS = "Which throw will the opponent make next?"


def state(throws: list[str]) -> str:
    return "The opponent's last throws, oldest first: " + ", ".join(throws[-MEMORY:]) + "."


def question() -> dict:
    return {"type": "choice", "instructions": INSTRUCTIONS, "criteria": THROWS}


def histories(count: int = 300, seed: int = 5) -> list[list[str]]:
    """Throws of players with a favourite: it is thrown three times in five, else any throw."""
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        favourite = rng.choice(THROWS)
        out.append([favourite if rng.random() < 0.6 else rng.choice(THROWS) for _ in range(rng.randint(3, 12))])
    return out
