"""Tic-Tac-Toe: what the model is asked, and why it is asked that way.

The game is in `webapp/static/tictactoe-core.js`; this module holds the same texts and logic
for `tools/tictactoe_check.py`, which runs every position through the board. The two must
stay the same.

Tic-Tac-Toe uses a general Laya checkpoint as it ships. The model is not shown the grid. For
each empty square the harness works out what taking it does and describes it with a word or
two, and those descriptions are the options of a `choice` question.

The wording was found on the board. The model takes "wins" reliably, but it does not rank
one good thing above another: offered "blocks a loss" next to "threatens to win" it took the
threat in a third of the cases, whatever the two were called, and it took "neutral" over
"threatens to win" or a losing square over a neutral one about as often. So a question never
offers more than one good description and one bad one. With a win to take, a loss to block or
a win to force, every other square is called "a mistake", which is simply what it is. Of the
words tried for the bad square, "a mistake" did best; "loses" and "misses the win" were taken
for good news one time in five.

How far the harness looks can be chosen. One move ahead it sees wins, blocks and two in a
row, and the player it describes for walks into forks: a perfect opponent beats it in a fifth
of the games it starts and four fifths of the others. Looking to the end of the game it knows
which squares force a win ("threatens to win"), hold the draw ("safe") or lose ("a mistake"),
and that player never loses.
"""
from functools import lru_cache

SQUARES = ["top-left", "top", "top-right", "left", "centre", "right", "bottom-left", "bottom", "bottom-right"]
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
STATE = "The player needs a move that wins, or blocks a loss."
INSTRUCTIONS = "Which square is the best move?"
# What taking a square does, best first.
OUTCOMES = ["wins", "blocks a loss", "threatens to win", "safe", "neutral", "a mistake"]
MISTAKE = 5


def line(board) -> bool:
    return any(board[a] and board[a] == board[b] == board[c] for a, b, c in LINES)


def other(mark: str) -> str:
    return "O" if mark == "X" else "X"


@lru_cache(maxsize=None)
def value(board: tuple, mark: str) -> int:
    """What a position is worth to the side to move with best play: 1, 0 or -1."""
    if line(board):
        return -1
    empty = [i for i in range(9) if not board[i]]
    if not empty:
        return 0
    return max(-value(board[:i] + (mark,) + board[i + 1:], other(mark)) for i in empty)


def outcomes(board: tuple, mark: str, deep: bool) -> dict[str, int]:
    """What each empty square does for `mark`: an index into OUTCOMES per square name."""
    empty = [i for i in range(9) if not board[i]]
    put = lambda i, who: board[:i] + (who,) + board[i + 1:]
    named = lambda level: {SQUARES[i]: level(i) for i in empty}
    wins = [i for i in empty if line(put(i, mark))]
    if wins:
        return named(lambda i: 0 if i in wins else MISTAKE)
    blocks = [i for i in empty if line(put(i, other(mark)))]
    if blocks:
        return named(lambda i: 1 if i in blocks else MISTAKE)
    if deep:
        worth = {i: -value(put(i, mark), other(mark)) for i in empty}
        if any(v > 0 for v in worth.values()):
            return named(lambda i: 2 if worth[i] > 0 else MISTAKE)
        return named(lambda i: 3 if worth[i] == 0 else MISTAKE)

    def threat(i):
        mine = put(i, mark)
        return any(i in ln and sum(mine[c] == mark for c in ln) == 2 and any(not mine[c] for c in ln) for ln in LINES)

    return named(lambda i: 2 if threat(i) else 4)


def question(levels: dict[str, int]) -> dict:
    return {"type": "choice", "instructions": INSTRUCTIONS,
            "criteria": {square: OUTCOMES[level] for square, level in levels.items()}}


def all_positions() -> list[tuple[tuple, str]]:
    """Every position the game can reach in which there is a move to make."""
    seen, out = set(), []

    def walk(board, mark):
        if board in seen or line(board) or all(board):
            return
        seen.add(board)
        out.append((board, mark))
        for i in range(9):
            if not board[i]:
                walk(board[:i] + (mark,) + board[i + 1:], other(mark))

    walk((None,) * 9, "X")
    return out
