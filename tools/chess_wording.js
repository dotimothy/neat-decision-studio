// Checks that the browser words a chess position and its moves exactly as the model's own
// engine does: the state, the list of legal moves and every question, string for string,
// against the dump of tools/chess_reference.py.
//
//   node tools/chess_wording.js [build/chess_reference.json]
const fs = require("fs"), path = require("path");
const { Chess } = require(path.join(__dirname, "..", "webapp/static/vendor/chess.js"));
const core = require(path.join(__dirname, "..", "webapp/static/chess-core.js"));

const reference = JSON.parse(fs.readFileSync(process.argv[2] || "build/chess_reference.json", "utf8"));
let questions = 0, wrong = 0;
const complain = (what, fen, got, want) => { if (wrong++ < 12) console.log(`${what} differs at ${fen}\n   here:   ${got}\n   engine: ${want}`); };
if (JSON.stringify(reference.criteria) !== JSON.stringify(core.CHESS_LEVELS)) complain("the levels", "-", core.CHESS_LEVELS, reference.criteria);
for (const position of reference.positions) {
  const game = new Chess(position.fen), moves = game.moves({ verbose: true });
  const state = JSON.stringify(core.chessState(game, moves));
  if (state !== JSON.stringify(position.state)) complain("the state", position.fen, state, JSON.stringify(position.state));
  const mine = new Map(moves.map((move) => [move.from + move.to + (move.promotion || ""), move]));
  if (mine.size !== position.moves.length) complain("the number of legal moves", position.fen, mine.size, position.moves.length);
  for (const wanted of position.moves) {
    questions++;
    const move = mine.get(wanted.uci);
    if (!move) { complain("a legal move", position.fen, "missing", wanted.uci); continue; }
    const asked = core.chessQuestion(game, move).instructions;
    if (asked !== wanted.instructions) complain("a question", position.fen, asked, wanted.instructions);
  }
}
console.log(`${reference.positions.length} positions, ${questions} questions: ${wrong ? wrong + " differences" : "worded exactly as the engine words them"}`);
process.exit(wrong ? 1 : 0);
