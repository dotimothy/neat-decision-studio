// Plays Pac-Man without a browser or a board, with the game code the page uses
// (webapp/static/pacman-core.js), taking a best-described move every time (the first of them,
// in the order up, left, down, right, when several are equally good), and for comparison a
// player that takes any open way at random. tools/pacman_check.py confirms on the board that
// the model picks a best-described move.
//
//   node tools/sim_pacman.js [games]
//
// It shows what the descriptions alone are worth, and which combinations of descriptions the
// game actually produces.
const path = require("path");
const g = require(path.join(__dirname, "..", "webapp/static/pacman-core.js"));

const games = Number(process.argv[2] || 200);

// The maze itself: rows of one length, the same mirrored, and no dead ends on the floor.
const maze = g.PACMAN_MAZE, world = new g.PacmanWorld(1);
for (const row of maze) {
  const plain = row.replace("P", ".");
  if (row.length !== maze[0].length) throw new Error(`maze row of ${row.length} squares: ${row}`);
  if (plain !== [...plain].reverse().join("")) throw new Error(`maze row not mirrored: ${row}`);
}
const reach = world.distances([world.start]);
for (let y = 0; y < maze.length; y++) for (let x = 0; x < maze[0].length; x++) {
  if (!world.open(x, y)) continue;
  if (reach[world.index({ x, y })] < 0) throw new Error(`square ${x},${y} cannot be reached`);
  const ways = g.PACMAN_MOVES.filter((move) => { const c = world.next({ x, y }, move); return world.open(c.x, c.y); }).length;
  if (ways < 2) throw new Error(`square ${x},${y} is a dead end`);
}

function play(pick) {
  const combos = new Map(), causes = new Map();
  let score = 0, levels = 0, steps = 0, best = 0, risky = 0, decisions = 0;
  for (let seed = 1; seed <= games; seed++) {
    const world = new g.PacmanWorld(seed * 7919);
    while (world.alive) {
      const { outcomes } = world.question();
      const key = g.PACMAN_MOVES.map((move) => move in outcomes ? outcomes[move] : "-").join("");
      combos.set(key, (combos.get(key) || 0) + 1);
      const move = pick(outcomes, world);
      decisions++; risky += outcomes[move] >= g.PACMAN_RISKY_FROM;
      world.step(move);
    }
    score += world.score; levels += world.cleared; steps += world.steps; best = Math.max(best, world.score);
    causes.set(world.cause, (causes.get(world.cause) || 0) + 1);
  }
  return { score: score / games, levels: levels / games, steps: steps / games, best, causes, combos, risky: risky / decisions };
}

const bestDescribed = (outcomes) => { const top = Math.min(...Object.values(outcomes)); return g.PACMAN_MOVES.find((move) => outcomes[move] === top); };
let state = 12345;
const atRandom = (outcomes) => { const moves = Object.keys(outcomes); state = (state * 1103515245 + 12345) & 0x7fffffff; return moves[state % moves.length]; };

for (const [name, pick] of [["best-described", bestDescribed], ["random", atRandom]]) {
  const r = play(pick);
  console.log(`${name}: ${games} games, ${r.score.toFixed(0)} points a game on average (best ${r.best}), `
    + `${r.levels.toFixed(2)} levels cleared a game, ${Math.round(r.steps)} steps a game; ${(100 * r.risky).toFixed(1)}% of moves dangerous or fatal`);
  console.log("  ended by:", Object.fromEntries(r.causes));
  if (pick === bestDescribed) {
    const total = [...r.combos.values()].reduce((a, b) => a + b, 0);
    console.log(`  ${r.combos.size} combinations of descriptions occurred (up, left, down, right; 0 best .. 3 fatal, - a wall):`);
    console.log("  " + [...r.combos.entries()].sort((a, b) => b[1] - a[1]).slice(0, 14).map(([k, n]) => `${k} ${(100 * n / total).toFixed(1)}%`).join("  "));
  }
}
