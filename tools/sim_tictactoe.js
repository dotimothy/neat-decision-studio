// Plays Tic-Tac-Toe without a browser or a board, with the game code the page uses
// (webapp/static/tictactoe-core.js): a player that always takes a best-described square (one
// of them at random when several are equally good) against a random player and against a
// perfect one. tools/tictactoe_check.py confirms on the board that the model takes a
// best-described square.
//
//   node tools/sim_tictactoe.js [games]
//
// It does so for both of the harness's ways of looking: one move ahead, and to the end of the game.
const path = require("path");
const g = require(path.join(__dirname, "..", "webapp/static/tictactoe-core.js"));
let seed = 12345;
const random = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
const pick = (items) => items[Math.floor(random() * items.length)];

let deep = false;
const described = (board, mark) => {
  const outcomes = g.tttOutcomes(board, mark, deep), best = Math.min(...Object.values(outcomes));
  return g.TTT_SQUARES.indexOf(pick(Object.keys(outcomes).filter((square) => outcomes[square] === best)));
};
const anySquare = (board) => pick(g.tttEmpty(board));
const value = g.tttValue;
const perfect = (board, mark) => {
  const scored = g.tttEmpty(board).map((i) => { board[i] = mark; const v = -value(board, g.tttOther(mark)); board[i] = null; return [i, v]; });
  const best = Math.max(...scored.map(([, v]) => v));
  return pick(scored.filter(([, v]) => v === best))[0];
};

const games = Number(process.argv[2] || 2000);
for (deep of [false, true]) {
console.log(deep ? "\nLooking to the end of the game:" : "Looking one move ahead:");
for (const [name, opponent] of [["a random player", anySquare], ["a perfect player", perfect], ["itself", described]]) {
  for (const first of [true, false]) {
    const tally = { win: 0, draw: 0, loss: 0 };
    for (let n = 0; n < games; n++) {
      const board = Array(9).fill(null), mine = first ? "X" : "O";
      let mark = "X";
      while (!g.tttLine(board) && g.tttEmpty(board).length) {
        board[mark === mine ? described(board, mark) : opponent(board, mark)] = mark;
        mark = g.tttOther(mark);
      }
      const line = g.tttLine(board);
      tally[!line ? "draw" : board[line[0]] === mine ? "win" : "loss"]++;
    }
    const pct = (n) => (100 * n / games).toFixed(1) + "%";
    console.log(`against ${name}, moving ${first ? "first " : "second"}: wins ${pct(tally.win)}, draws ${pct(tally.draw)}, loses ${pct(tally.loss)}`);
  }
}
}
