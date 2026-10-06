// Plays Snake without a browser or a board, with the game code the page uses
// (webapp/static/snake-core.js), taking a best-described move every time (the first of them
// when several are equally good). tools/snake_check.py confirms on the board that the model
// picks a best-described move.
//
//   node tools/sim_snake.js [games] [columns] [rows] [sight]
//
// `sight` is how many squares around its head the snake sees of its own body (default: all).
//
// It shows what the descriptions alone are worth, and which combinations of descriptions the
// game actually produces.
const path = require("path");
const g = require(path.join(__dirname, "..", "webapp/static/snake-core.js"));

const games = Number(process.argv[2] || 200);
const grid = { cols: Number(process.argv[3] || g.SNAKE_GRID.cols), rows: Number(process.argv[4] || g.SNAKE_GRID.rows) };
const combos = new Map(), causes = new Map();
const sight = process.argv[5] ? Number(process.argv[5]) : Infinity;
let food = 0, steps = 0, best = 0, size = "";
for (let seed = 1; seed <= games; seed++) {
  const world = new g.SnakeWorld(seed * 7919, grid, sight);
  size = `${world.cols}x${world.rows}`;
  while (world.alive) {
    const { outcomes } = world.question();
    const key = g.SNAKE_MOVES.map((move) => outcomes[move]).join("");
    combos.set(key, (combos.get(key) || 0) + 1);
    const top = Math.min(...Object.values(outcomes));
    world.step(g.SNAKE_MOVES.find((move) => outcomes[move] === top));
  }
  food += world.eaten; steps += world.steps; best = Math.max(best, world.eaten);
  causes.set(world.cause, (causes.get(world.cause) || 0) + 1);
}
console.log(`${games} games on ${size}, seeing ${sight === Infinity ? "the whole board" : sight + " around the head"}: ${(food / games).toFixed(1)} food per game on average, best ${best}, ${Math.round(steps / games)} moves per game`);
console.log("ended by:", Object.fromEntries(causes));
const total = [...combos.values()].reduce((a, b) => a + b, 0);
console.log(`${combos.size} of 64 combinations of descriptions occurred (left, straight, right; 0 best .. 3 blocked):`);
console.log([...combos.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12).map(([k, n]) => `${k} ${(100 * n / total).toFixed(1)}%`).join("  "));
