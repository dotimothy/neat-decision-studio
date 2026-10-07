// Checks how the chess harness keeps a player from repeating positions, without a browser or
// a board. The model is not here, so a stand-in rates the moves: the material left after the
// move, plus a fixed preference per position and move. Like the model it has no memory of the
// game, and two of it shuffle pieces until the game is drawn by repetition. The games are
// played twice, once choosing by rating alone and once with the harness's rule
// (webapp/static/chess-core.js: chessSeen, chessWorth, chessCandidates), and the endings counted.
//
//   node tools/sim_chess.js [games] [plies]
const path = require("path");
const { Chess } = require(path.join(__dirname, "..", "webapp/static/vendor/chess.js"));
const core = require(path.join(__dirname, "..", "webapp/static/chess-core.js"));

const WORTH = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
const hash = (text) => { let h = 2166136261; for (let i = 0; i < text.length; i++) { h ^= text.charCodeAt(i); h = Math.imul(h, 16777619); } return (h >>> 0) / 4294967296; };
// A win chance for the side that makes `move`: who has more material afterwards, and a taste for certain moves.
function rate(game, move, salt) {
  const mover = game.turn();
  game.move(move);
  let balance = 0;
  for (const piece of game.board().flat()) if (piece) balance += (piece.color === mover ? 1 : -1) * WORTH[piece.type];
  const key = core.chessKey(game);
  game.undo();
  return 1 / (1 + Math.exp(-0.4 * balance)) + 0.04 * (hash(key + move.san + salt) - 0.5);
}

function play(salt, avoid, plies) {
  const game = new Chess();
  let repeats = 0;
  while (!game.game_over() && game.history().length < plies) {
    const seen = core.chessSeen(game);
    const rated = game.moves({ verbose: true }).map((move) => ({ move, exact: core.chessExact(game, move), seen: core.chessAfter(game, move, seen) }));
    for (const item of rated) item.win = item.exact ?? rate(game, item.move, salt);
    let chosen;
    if (avoid) {
      const playable = core.chessCandidates(rated);
      chosen = playable.reduce((best, item) => (core.chessWorth(item) > core.chessWorth(best) ? item : best));
    } else {
      chosen = rated.reduce((best, item) => (item.win > best.win ? item : best));
    }
    repeats += chosen.seen > 0;
    game.move(chosen.move);
  }
  const how = game.in_checkmate() ? "checkmate" : game.in_threefold_repetition() ? "threefold repetition" : game.in_stalemate() ? "stalemate"
    : game.insufficient_material() ? "insufficient material" : game.in_draw() ? "the fifty-move rule" : `still going after ${plies} plies`;
  return { how, plies: game.history().length, repeats };
}

const games = Number(process.argv[2] || 30), plies = Number(process.argv[3] || 300);
for (const avoid of [false, true]) {
  const endings = {}; let moves = 0, repeats = 0;
  for (let n = 0; n < games; n++) {
    const result = play("game" + n, avoid, plies);
    endings[result.how] = (endings[result.how] || 0) + 1; moves += result.plies; repeats += result.repeats;
  }
  console.log(`${avoid ? "with the harness's rule" : "choosing by rating alone"}: ${games} games, ${Math.round(moves / games)} plies each, `
    + `${(repeats / games).toFixed(1)} moves a game back into an earlier position`);
  console.log("   ended by:", endings);
}
