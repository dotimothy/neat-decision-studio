// Chess: how a position and a move are put to the chess model, and how its answers pick a
// move. Loaded by chess.html and by tools/chess_wording.js; the rules of
// chess themselves are chess.js (vendor/chess.js, BSD-2-Clause).
//
// The model is LayaChess (datafreak/laya-chess): Laya fine-tuned on Stockfish-rated moves. It
// answers one `score` question per legal move, "what is the side to move's win chance after
// this move?", on ten levels. So a move takes one decision per legal move, about thirty of
// them, and the move played is the one with the highest win chance. The wording below is the
// one the model was trained on (its chess_meta.json, and the "piece lists v2" state its engine
// builds); tools/chess_wording.js checks it against the engine's own, string for string.
"use strict";
if (typeof Chess === "undefined" && typeof require !== "undefined") { var Chess = require("./vendor/chess.js").Chess; }    // node: the page loads it itself

const CHESS_LEVELS = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"];
const CHESS_PIECES = { k: "king", q: "queen", r: "rook", b: "bishop", n: "knight", p: "pawn" };
const CHESS_ORDER = "kqrbnp";
const CHESS_FILES = "abcdefgh";

// One side's pieces, kings first, each kind from a1 along the ranks to h8: "Ke1 Qd1 Ra1 Rh1 ...".
function chessPieceList(game, color) {
  const board = game.board(), out = [];          // board[0] is rank 8
  for (const type of CHESS_ORDER) {
    for (let rank = 1; rank <= 8; rank++) {
      for (let file = 0; file < 8; file++) {
        const piece = board[8 - rank][file];
        if (piece && piece.type === type && piece.color === color) out.push(type.toUpperCase() + CHESS_FILES[file] + rank);
      }
    }
  }
  return out.join(" ");
}

// The state every question about a position shares. `moves` is the position's legal moves
// (chess.js, verbose), passed in to avoid generating them twice.
function chessState(game, moves) {
  const fen = game.fen().split(" ");
  return { to_move: game.turn() === "w" ? "white" : "black", white: chessPieceList(game, "w"), black: chessPieceList(game, "b"),
           castling: fen[2],
           // Only an en passant capture that can actually be played is mentioned.
           en_passant: moves.some((move) => move.flags.includes("e")) ? fen[3] : "-" };
}

// The question about one move: "white plays Nf3 (knight g1-f3). Win chance for white?"
function chessQuestion(game, move) {
  const side = game.turn() === "w" ? "white" : "black";
  let extra = "";
  if (move.captured) extra += `, takes ${CHESS_PIECES[move.captured]}`;
  if (move.promotion) extra += `, promotes to ${CHESS_PIECES[move.promotion]}`;
  if (/[+#]$/.test(move.san)) extra += ", check";
  return { type: "score", criteria: CHESS_LEVELS,
           instructions: `${side} plays ${move.san} (${CHESS_PIECES[move.piece]} ${move.from}-${move.to}${extra}). Win chance for ${side}?` };
}

// The win chance in a `score` answer: the levels' midpoints weighted by their probabilities.
// The runtime's `score` is the expected level, so that is (score + 0.5) / 10.
function chessWinChance(answer) { return (answer.score + 0.5) / CHESS_LEVELS.length; }

// What the rules alone say about a move: 1 if it mates, 0.5 if it ends the game drawn by
// stalemate, bare kings or the fifty-move rule, else null. These are never asked of the
// model. A draw by repetition is not among them: repeating is something the player is kept
// from, below, not an outcome it weighs.
function chessExact(game, move) {
  game.move(move);
  const value = game.in_checkmate() ? 1
    : game.in_stalemate() || game.insufficient_material() || (game.in_draw() && !game.in_threefold_repetition()) ? 0.5 : null;
  game.undo();
  return value;
}

// Repetition. A model that rates each position on its own has no sense of having been there
// before, and two of them shuffle the same pieces back and forth until the game is drawn by
// threefold repetition. So the harness remembers for it: `chessSeen` counts how often each
// position has stood on the board, and a move is described by how often the position it
// leads to has (`seen`). A move back into a position is marked down, and one that would bring
// a position about for the third time is not played while any other move exists.
const CHESS_REPEAT_PENALTY = 0.05;       // of win chance, for going back into an earlier position
const chessKey = (game) => game.fen().split(" ").slice(0, 4).join(" ");      // what "the same position" means
function chessSeen(game) {
  const replay = new Chess(), seen = new Map([[chessKey(replay), 1]]);
  // A game that did not start from the usual position is counted from where it stands.
  if (game.history().length === 0 || game.header().FEN) return new Map([[chessKey(game), 1]]);
  for (const san of game.history()) { replay.move(san); seen.set(chessKey(replay), (seen.get(chessKey(replay)) || 0) + 1); }
  return seen;
}
function chessAfter(game, move, seen) {
  game.move(move);
  const count = seen.get(chessKey(game)) || 0;
  game.undo();
  return count;
}
// What a rated move is worth when choosing: its win chance (after a second look, if it had
// one), less the mark-down for repeating.
const chessWorth = (item) => (item.second ?? item.win) - (item.seen ? CHESS_REPEAT_PENALTY : 0);
// The moves to choose among: all but those that would repeat a position for the third time,
// unless nothing else is legal.
function chessCandidates(rated) {
  const fresh = rated.filter((item) => (item.seen || 0) < 2);
  return fresh.length ? fresh : rated;
}

// How a finished game ended, in words; null while it goes on.
function chessResult(game) {
  if (!game.game_over()) return null;
  if (game.in_checkmate()) return { winner: game.turn() === "w" ? "b" : "w", how: "checkmate" };
  return { winner: null, how: game.in_stalemate() ? "stalemate" : game.in_threefold_repetition() ? "threefold repetition"
                                : game.insufficient_material() ? "insufficient material" : "the fifty-move rule" };
}

if (typeof module !== "undefined") {
  module.exports = { CHESS_LEVELS, CHESS_PIECES, chessPieceList, chessState, chessQuestion, chessWinChance, chessExact, chessResult,
                     CHESS_REPEAT_PENALTY, chessKey, chessSeen, chessAfter, chessWorth, chessCandidates };
}
