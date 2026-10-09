// A traditional chess engine for the models to play against: alpha-beta search, in JavaScript.
//
// It is the classical recipe and nothing learned. A position is scored by counting: what
// each side's pieces are worth, and where they stand. A move is chosen by looking ahead:
// negamax (minimax with the sign flipped each ply) with alpha-beta pruning, which stops
// looking at a move as soon as it is known to be worse than one already found. Around that,
// the usual three helpers:
//
//   iterative deepening   depth 1, then 2, ... up to the depth asked for or the time allowed;
//                         each depth's best move is tried first at the next
//   move ordering         captures of the most valuable piece by the least valuable first,
//                         then promotions: good moves early are what make the pruning cut
//   quiescence            at the end of the look-ahead, captures are played out before the
//                         position is scored, so it is not scored in the middle of an exchange
//
// The rules come from chess.js (vendor/chess.js), which is not fast: this searches three to
// five thousand positions a second, so three plies take about a second and four several. It runs in a worker, so
// the page stays alive while it thinks:
//
//   const worker = new Worker("/static/chess-engine.js");
//   worker.postMessage({ id, moves: ["e4", "e5"], depth: 3, ms: 4000 });      // the game so far, in SAN
//   worker.onmessage = ({ data }) => data.move, data.score, data.depth, data.nodes, data.pruned, data.ms, data.line
//
// `score` is in pawns for the side to move, `pruned` the branches alpha-beta cut off. The
// answer also says how the search went, for a page to show: `depths`, what each depth of the
// deepening found; `root`, every first move it weighed at the last depth, in the order it
// tried them, each with its score or the most it could be worth and the positions spent on
// it; `quiet`, how many of the positions were in the playing out of captures; and `counted`,
// the position's material and placement for each side, which is all the judgement it has.
"use strict";

const VALUE = { p: 100, n: 320, b: 330, r: 500, q: 900, k: 0 };
const MATE = 100000;

// Where a piece stands, in hundredths of a pawn, for White on rank r (0 = its own back rank)
// and file f. Written as rules, not tables: the centre for knights and bishops, advancing for
// pawns, the seventh rank and open centre files for rooks, and for the king its corner while
// the queens are on and the centre once they are off.
function placement(type, r, f, endgame) {
  const centre = 3.5 - Math.max(Math.abs(f - 3.5), Math.abs(r - 3.5));      // 0 at the edge, 3 in the middle
  const middleFile = 3.5 - Math.abs(f - 3.5);
  switch (type) {
    case "p": return r * 6 + (f >= 2 && f <= 5 ? (r >= 3 ? 14 : r === 1 ? -12 : 4) : 0) + (r === 6 ? 40 : 0);
    case "n": return centre * 12 - 22;
    case "b": return centre * 7 - 8 - (r === 0 ? 8 : 0);
    case "r": return (r === 6 ? 18 : 0) + middleFile * 3;
    case "q": return centre * 3 - (r === 0 ? 4 : 0);
    case "k": return endgame ? centre * 14 - 20 : (r === 0 ? 14 : -20 * r) + (f <= 2 || f >= 6 ? 16 : -10);
    default: return 0;
  }
}

// The position for the side to move: its material and placement less the other side's.
// With `parts`, the four sums it is made of are written there (material and placement, for
// White and for Black) and whether the endgame's king placement was used.
function evaluate(game, parts) {
  const board = game.board();
  let queens = 0, minor = 0;
  for (const row of board) for (const piece of row) if (piece) { if (piece.type === "q") queens++; else if (piece.type !== "p" && piece.type !== "k") minor++; }
  const endgame = queens === 0 || minor <= 2;
  let score = 0;
  if (parts) Object.assign(parts, { material: { w: 0, b: 0 }, placement: { w: 0, b: 0 }, endgame });
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      const piece = board[y][x];
      if (!piece) continue;
      const rank = piece.color === "w" ? 7 - y : y;                         // from its own back rank
      const where = placement(piece.type, rank, x, endgame);
      if (parts) { parts.material[piece.color] += VALUE[piece.type]; parts.placement[piece.color] += where; }
      score += piece.color === "w" ? VALUE[piece.type] + where : -VALUE[piece.type] - where;
    }
  }
  return game.turn() === "w" ? score : -score;
}

// Captures of the most valuable piece by the least valuable first, then promotions.
const order = (move) => (move.captured ? 10 * VALUE[move.captured] - VALUE[move.piece] + 10000 : 0) + (move.promotion ? 9000 : 0);

let nodes = 0, quiet = 0, pruned = 0, deadline = 0, stopped = false;

// Captures played out until the position is quiet, so that it is not scored mid-exchange.
function quiesce(game, alpha, beta, depth) {
  nodes++; quiet++;
  const stand = evaluate(game);
  if (stand >= beta) return beta;
  if (stand > alpha) alpha = stand;
  if (depth === 0) return alpha;
  const captures = game.moves({ verbose: true }).filter((move) => move.captured || move.promotion).sort((a, b) => order(b) - order(a));
  for (const move of captures) {
    game.move(move);
    const score = -quiesce(game, -beta, -alpha, depth - 1);
    game.undo();
    if (score >= beta) { pruned++; return beta; }
    if (score > alpha) alpha = score;
  }
  return alpha;
}

// Negamax with alpha-beta: the best score the side to move can be sure of, within (alpha, beta).
function search(game, depth, alpha, beta, ply, line) {
  if ((nodes & 63) === 0 && performance.now() > deadline) stopped = true;
  if (stopped) return 0;
  const moves = game.moves({ verbose: true });
  if (!moves.length) { nodes++; return game.in_check() ? -MATE + ply : 0; }           // mated, sooner is worse; or stalemate
  // A draw by repetition or by too little material is looked for only right after the engine's
  // own move: chess.js replays the whole game to answer, which is too slow to ask at every node.
  if (ply === 1 && (game.insufficient_material() || game.in_threefold_repetition())) { nodes++; return 0; }
  if (depth === 0) return quiesce(game, alpha, beta, 4);
  nodes++;
  moves.sort((a, b) => order(b) - order(a));
  const first = line[ply];                                                     // the last depth's best move here
  if (first) { const at = moves.findIndex((move) => move.san === first); if (at > 0) moves.unshift(moves.splice(at, 1)[0]); }
  let best = -Infinity;
  for (const move of moves) {
    game.move(move);
    const below = move.san === first ? line.slice() : line.slice(0, ply + 1);  // along the last depth's line, its moves are tried first
    const score = -search(game, depth - 1, -beta, -alpha, ply + 1, below);
    game.undo();
    if (stopped) return 0;
    if (score > best) {
      best = score;
      if (score > alpha) {
        alpha = score;
        line.length = ply; line[ply] = move.san;
        for (let i = ply + 1; i < below.length; i++) line[i] = below[i];
      }
    }
    if (alpha >= beta) { pruned++; break; }                                    // the other side will not allow this line
  }
  return best;
}

// The move for the side to move after `moves` (the game so far): deeper and deeper until the
// depth asked for is done or the time is up. A depth that was cut short is not used.
//
// The first moves are gone through here and not in `search`, so that what happened to each
// can be told: the best so far sets the bar (alpha), and a later move is searched only far
// enough to show it does not clear the bar, which gives the most it could be worth and not
// its score. They are tried in the order the depth before ranked them.
function choose(sans, depth, ms) {
  const game = new Chess();
  for (const san of sans) game.move(san);
  const started = performance.now();
  nodes = quiet = pruned = 0; stopped = false; deadline = started + ms;
  const counted = {};
  evaluate(game, counted);
  let legal = game.moves({ verbose: true }).sort((a, b) => order(b) - order(a));
  let best = { move: legal[Math.floor(Math.random() * legal.length)], score: 0, depth: 0, line: [] };
  let line = [], root = [];
  const depths = [];
  for (let d = 1; d <= depth; d++) {
    const before = { nodes, pruned, at: performance.now() }, weighed = [];
    let alpha = -Infinity, found = null, foundLine = [];
    for (const move of legal) {
      const spent = nodes, below = move.san === line[0] ? line.slice() : [move.san];
      game.move(move);
      const score = -search(game, d - 1, -Infinity, -alpha, 1, below);
      game.undo();
      if (stopped) break;
      const exact = score > alpha;                       // it cleared the bar: this is its score
      if (exact) { alpha = score; found = move; foundLine = [move.san, ...below.slice(1)]; }
      weighed.push({ san: move.san, score: score / 100, exact, mate: mateIn(score), nodes: nodes - spent });
    }
    if (stopped) break;
    line = foundLine; root = weighed;
    best = { move: found, score: alpha, depth: d, line: line.slice() };
    depths.push({ depth: d, move: found.san, score: alpha / 100, mate: mateIn(alpha), nodes: nodes - before.nodes, pruned: pruned - before.pruned,
                  ms: performance.now() - before.at });
    // The next depth tries this depth's best first, then the rest as this depth left them.
    const rank = new Map(weighed.map((entry, at) => [entry.san, entry.san === found.san ? -1 : at]));
    legal = legal.slice().sort((a, b) => rank.get(a.san) - rank.get(b.san));
    if (Math.abs(alpha) > MATE - 100) break;                                    // a forced mate is found: no need to look further
  }
  return { ...best, nodes, quiet, pruned, ms: performance.now() - started, depths, root, counted, cut: stopped };
}

// Moves to mate for a score that is one (positive: the side to move mates), else null.
const mateIn = (score) => Math.abs(score) > MATE - 100 ? Math.sign(score) * Math.ceil((MATE - Math.abs(score)) / 2) : null;

if (typeof importScripts === "function") {                                      // as a worker
  importScripts("/static/vendor/chess.js");
  onmessage = ({ data }) => {
    const found = choose(data.moves, data.depth, data.ms);
    postMessage({ id: data.id, move: { from: found.move.from, to: found.move.to, promotion: found.move.promotion, san: found.move.san },
                  score: found.score / 100, mate: mateIn(found.score), depth: found.depth, asked: data.depth, cut: found.cut,
                  nodes: found.nodes, quiet: found.quiet, pruned: found.pruned, ms: found.ms, line: found.line,
                  depths: found.depths, root: found.root, counted: found.counted });
  };
} else if (typeof module !== "undefined") {                                     // under node, for tools/
  module.exports = { choose, evaluate, setChess: (constructor) => { globalThis.Chess = constructor; } };
}
