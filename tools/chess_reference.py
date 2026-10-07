#!/usr/bin/env python3
"""PyTorch reference for the chess model: positions, the question for every legal move, and
the win chance upstream's engine gets for it.

Runs in `.venv-ref` (torch + transformers + upstream laya) with python-chess installed and the
LayaChess engine cloned to third_party/LayaChess; it uses that engine's own encoding and
scoring, so the dump is what the engine itself would compute, in fp32 on the CPU.

    .venv-ref/bin/python tools/chess_reference.py --out build/chess_reference.json

The dump is what `tools/chess_check.py` compares the board against, and what
`tools/chess_wording.js` compares the browser's wording of a position against.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "LayaChess" / "engine"))
from laya_chess.encoding import board_state          # noqa: E402
from laya_chess.model import LayaChessModel           # noqa: E402


def positions(games: int, seed: int) -> list[chess.Board]:
    """Positions from random games, sampled from the opening to the endgame, plus a few fixed
    ones that exercise castling, en passant, promotion and check."""
    rng = random.Random(seed)
    out = [chess.Board()]
    for _ in range(games):
        board = chess.Board()
        for ply in range(140):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
            if ply in (3, 9, 17, 29, 45, 69, 99, 139):
                out.append(board.copy())
    for fen in ["r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 6 9",       # both may castle
                "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",               # en passant
                "8/P4k2/8/8/8/8/5K1p/8 w - - 0 1",                                             # promotions
                "8/8/8/8/8/5k2/4q3/6K1 w - - 0 1",                                             # in check
                "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"]:                                       # mate in one
        out.append(chess.Board(fen))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/laya-chess")
    ap.add_argument("--out", default="build/chess_reference.json")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    model = LayaChessModel(args.model, device="cpu")
    dump, longest = [], 0
    for board in positions(args.games, args.seed):
        if board.is_game_over():
            continue
        moves, win = model.evaluate([board])[0]
        state = board_state(board)
        entry = {"fen": board.fen(), "state": state, "moves": []}
        for move, value in zip(moves, win):
            question = model.encoder.question(board, move)
            ids, _ = model.encoder.encode(board, move, state)
            longest = max(longest, len(ids))
            entry["moves"].append({"uci": move.uci(), "san": board.san(move), "instructions": question["ins"],
                                   "tokens": len(ids), "win": round(float(value), 5)})
        dump.append(entry)
        best = max(entry["moves"], key=lambda m: m["win"])
        print(f"{board.fen():68s} {len(moves):2d} moves, up to {max(m['tokens'] for m in entry['moves'])} tokens, "
              f"best {best['san']} {best['win']:.2f}", flush=True)
    Path(args.out).write_text(json.dumps({"criteria": model.encoder.crit, "positions": dump}, indent=1))
    print(f"wrote {args.out}: {len(dump)} positions, {sum(len(p['moves']) for p in dump)} questions, longest {longest} tokens")


if __name__ == "__main__":
    main()
