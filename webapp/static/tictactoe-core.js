// Tic-Tac-Toe: the game, and how the harness describes a position to the model. Loaded by
// tictactoe.html and tools/sim_tictactoe.js; the texts are repeated in
// games/tictactoe_policy.py, which explains the wording, and must stay the same there.
//
// The model is not shown the grid. For each empty square the harness works out what taking it
// does (completes a line, stops the opponent's, makes two in a row) and says so in a word or
// two. Those descriptions are the options of a `choice` question, and the model picks one.
//
// How far the harness looks is `deep`. Looking one move ahead it sees a win to take, a loss to
// block and two in a row, and calls every other square neutral, so the player it describes
// for walks into a fork. Looking to the end of the game it knows what each square leads to
// with best play: one that forces a win "threatens to win", one that holds the draw is "safe",
// and one that loses is "a mistake".
"use strict";

const TTT_SQUARES = ["top-left", "top", "top-right", "left", "centre", "right", "bottom-left", "bottom", "bottom-right"];
const TTT_LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
const TTT_STATE = "The player needs a move that wins, or blocks a loss.";
const TTT_INSTRUCTIONS = "Which square is the best move?";
// What taking a square does, best first. The model reads these exact words. A question never
// offers more than one good description and one bad one: with a win to take, a loss to block
// or a win to force, every other square is "a mistake", which is simply what it is.
const TTT_OUTCOMES = ["wins", "blocks a loss", "threatens to win", "safe", "neutral", "a mistake"];
const TTT_MISTAKE = 5;

// The winning line of a board (nine cells of "X", "O" or null), or null.
function tttLine(board) {
  return TTT_LINES.find(([a, b, c]) => board[a] && board[a] === board[b] && board[a] === board[c]) || null;
}
const tttOther = (mark) => (mark === "X" ? "O" : "X");
const tttEmpty = (board) => board.flatMap((cell, i) => (cell ? [] : [i]));

// What a position is worth to the side to move with best play: 1 a win, 0 a draw, -1 a loss.
const tttValues = new Map();
function tttValue(board, mark) {
  const key = board.map((cell) => cell || "-").join("") + mark;
  if (!tttValues.has(key)) {
    let best = -1;                                           // a finished line: the side to move has lost
    if (!tttLine(board)) {
      const empty = tttEmpty(board);
      if (!empty.length) best = 0;
      else for (const i of empty) { const next = board.slice(); next[i] = mark; best = Math.max(best, -tttValue(next, tttOther(mark))); }
    }
    tttValues.set(key, best);
  }
  return tttValues.get(key);
}

// What each empty square does for `mark`: an index into TTT_OUTCOMES per square name.
function tttOutcomes(board, mark, deep = false) {
  const with_ = (i, who) => { const next = board.slice(); next[i] = who; return next; };
  const empty = tttEmpty(board), named = (level) => Object.fromEntries(empty.map((i) => [TTT_SQUARES[i], level(i)]));
  const wins = empty.filter((i) => tttLine(with_(i, mark)));
  if (wins.length) return named((i) => (wins.includes(i) ? 0 : TTT_MISTAKE));
  const blocks = empty.filter((i) => tttLine(with_(i, tttOther(mark))));
  if (blocks.length) return named((i) => (blocks.includes(i) ? 1 : TTT_MISTAKE));
  if (deep) {
    const worth = new Map(empty.map((i) => [i, -tttValue(with_(i, mark), tttOther(mark))]));
    const forced = empty.some((i) => worth.get(i) > 0);
    return named((i) => (forced ? (worth.get(i) > 0 ? 2 : TTT_MISTAKE) : worth.get(i) === 0 ? 3 : TTT_MISTAKE));
  }
  return named((i) => {
    const next = with_(i, mark);
    return TTT_LINES.some((line) => line.includes(i) && line.filter((c) => next[c] === mark).length === 2 && line.some((c) => !next[c])) ? 2 : 4;
  });
}

// The question for a position: the options are the empty squares, described.
function tttQuestion(board, mark, deep = false) {
  const outcomes = tttOutcomes(board, mark, deep);
  return { outcomes, question: { type: "choice", instructions: TTT_INSTRUCTIONS,
    criteria: Object.fromEntries(Object.entries(outcomes).map(([square, level]) => [square, TTT_OUTCOMES[level]])) } };
}

if (typeof module !== "undefined") {
  module.exports = { TTT_SQUARES, TTT_LINES, TTT_STATE, TTT_INSTRUCTIONS, TTT_OUTCOMES, TTT_MISTAKE, tttLine, tttOther, tttEmpty, tttValue, tttOutcomes, tttQuestion };
}
