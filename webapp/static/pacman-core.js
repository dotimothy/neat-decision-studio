// Pac-Man: the game, and how the harness describes a position to the model. Loaded by
// pacman.html and by tools/sim_pacman.js; the texts are repeated in games/pacman_policy.py,
// which explains the wording, and must stay the same there.
//
// The model is not shown the maze. For each way Pac-Man can go from where it stands, the
// harness works out what that leads to (does a ghost catch it on the next step, does it close
// in on a ghost nearby, does it get closer to the food: a blue ghost it can still catch, or
// else the nearest dot) and says so in a few words, in the words that worked for Snake. Those
// descriptions are the options of a `choice` question, and the model picks one.
//
// The game is played in steps: Pac-Man moves one square a step. The ghosts start a little
// slower and get faster with every level, until they are faster than Pac-Man; in the tunnel
// and while blue they go at half their pace.
"use strict";

// The maze, drawn for this game. # wall, . dot, o power pellet, - the door of the ghosts'
// house, G inside the house, P where Pac-Man starts, a space open floor. A row open at both
// ends is a tunnel from one side to the other.
const PACMAN_MAZE = [
  "###################",
  "#........#........#",
  "#o##.###.#.###.##o#",
  "#.................#",
  "#.##.#.#####.#.##.#",
  "#....#...#...#....#",
  "####.###.#.###.####",
  "####.#       #.####",
  "####.# ##-## #.####",
  "    .  #GGG#  .    ",
  "####.# ##### #.####",
  "####.#       #.####",
  "####.# ##### #.####",
  "#........#........#",
  "#.##.###.#.###.##.#",
  "#o.#.....P.....#.o#",
  "##.#.#.#####.#.#.##",
  "#....#...#...#....#",
  "#.######.#.######.#",
  "#.................#",
  "###################",
];
const PACMAN_MOVES = ["up", "left", "down", "right"];
const PACMAN_STEP = { up: { x: 0, y: -1 }, left: { x: -1, y: 0 }, down: { x: 0, y: 1 }, right: { x: 1, y: 0 } };
const PACMAN_REVERSE = { up: "down", down: "up", left: "right", right: "left" };
// The options' names in the question. Plain "right" reads as "correct": offered "down" toward
// the food and "right" away from it, the general model took "right" every time.
const PACMAN_OPTIONS = { up: "go up", left: "go left", down: "go down", right: "go right" };
const pacmanMove = (option) => PACMAN_MOVES.find((move) => PACMAN_OPTIONS[move] === option);
const PACMAN_STATE = "Pac-Man needs a safe move that brings it closer to the food.";
const PACMAN_INSTRUCTIONS = "Which way should Pac-Man go?";
// What a move can lead to, best first. The model reads these exact words.
const PACMAN_OUTCOMES = ["safe and toward the food", "safe but away from the food", "a ghost is near, dangerous",
                         "caught by a ghost, Pac-Man dies"];
const PACMAN_RISKY_FROM = 2;         // outcomes from this index on put Pac-Man in danger
const PACMAN_DANGER = 4;             // a ghost this many squares away or nearer, and getting nearer, is a danger
const PACMAN_LIVES = 3;
// The ghosts: who they are, where they start, the corner each heads for when scattering, and
// how many steps after a start each waits in the house.
const PACMAN_GHOSTS = [
  { name: "Blinky", color: "#e5484d", x: 9, y: 7, corner: { x: 17, y: -2 }, release: 0 },
  { name: "Pinky", color: "#f38fbf", x: 9, y: 9, corner: { x: 1, y: -2 }, release: 0 },
  { name: "Inky", color: "#3ec7d6", x: 8, y: 9, corner: { x: 18, y: 21 }, release: 24 },
  { name: "Clyde", color: "#f59f3a", x: 10, y: 9, corner: { x: 0, y: 21 }, release: 64 },
];
// Scatter and chase take turns, in steps; frightened time does not count.
const PACMAN_MODES = [["scatter", 56], ["chase", 160], ["scatter", 56], ["chase", 160], ["scatter", 40], ["chase", 160],
                      ["scatter", 40], ["chase", Infinity]];
// The ghosts' pace by level, in squares a step (the last holds from then on); Blinky goes faster
// still when few dots are left, as in the arcade.
const PACMAN_PACE = [0.8, 0.9, 1.0, 1.05, 1.1];
const PACMAN_HUNGER = 400;           // steps without a dot after which a player is taken to have stopped playing

class PacmanWorld {
  constructor(seed) {
    this.cols = PACMAN_MAZE[0].length; this.rows = PACMAN_MAZE.length;
    const find = (mark) => { const y = PACMAN_MAZE.findIndex((row) => row.includes(mark)); return { x: PACMAN_MAZE[y].indexOf(mark), y }; };
    this.start = find("P"); this.door = find("-");
    this.reset(seed);
  }

  // A new game: three lives, level 1, every dot.
  reset(seed) {
    this.seed = seed | 0;
    this.score = 0; this.lives = PACMAN_LIVES; this.level = 1; this.cleared = 0;
    this.alive = true; this.cause = ""; this.steps = 0; this.event = "";
    this.fill();
    this.restart();
  }

  random() {
    this.seed = this.seed + 0x6D2B79F5 | 0;
    let t = Math.imul(this.seed ^ this.seed >>> 15, 1 | this.seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }

  // The dots of a level: 1 a dot, 2 a power pellet.
  fill() {
    this.dots = new Uint8Array(this.cols * this.rows);
    this.left = 0;
    PACMAN_MAZE.forEach((row, y) => [...row].forEach((mark, x) => {
      if (mark === "." || mark === "o") { this.dots[y * this.cols + x] = mark === "o" ? 2 : 1; this.left++; }
    }));
  }

  // Everyone back to the start, after a life is lost or a level is cleared; the dots stay as they are.
  restart() {
    this.pac = { ...this.start }; this.heading = "left";
    this.ghosts = PACMAN_GHOSTS.map((ghost) => ({ name: ghost.name, color: ghost.color, corner: ghost.corner,
      x: ghost.x, y: ghost.y, px: ghost.x, py: ghost.y, dir: "left", scared: false, credit: 0,
      state: ghost.y === this.door.y - 1 ? "out" : "home", release: ghost.release }));
    this.tick = 0; this.frightened = 0; this.combo = 0; this.hungry = 0;
    this.mode = 0; this.modeTicks = 0;
  }

  // A copy to try a move on.
  clone() {
    const copy = Object.assign(Object.create(PacmanWorld.prototype), this);
    copy.pac = { ...this.pac }; copy.dots = this.dots.slice();
    copy.ghosts = this.ghosts.map((ghost) => ({ ...ghost }));
    return copy;
  }

  index(cell) { return cell.y * this.cols + cell.x; }
  wrap(x) { return (x + this.cols) % this.cols; }
  mark(x, y) { return y < 0 || y >= this.rows ? "#" : PACMAN_MAZE[y][this.wrap(x)]; }
  // Floor Pac-Man and a ghost out of the house may stand on.
  open(x, y) { const mark = this.mark(x, y); return mark !== "#" && mark !== "-" && mark !== "G"; }
  next(cell, move) { const d = PACMAN_STEP[move]; return { x: this.wrap(cell.x + d.x), y: cell.y + d.y }; }
  canGo(move) { const cell = this.next(this.pac, move); return this.open(cell.x, cell.y); }
  get chasing() { return PACMAN_MODES[this.mode][0] === "chase"; }

  // Steps to every square from the `starts`, along the floor (-1 where there is no way).
  distances(starts) {
    const dist = new Int16Array(this.cols * this.rows).fill(-1), queue = [];
    for (const cell of starts) if (dist[this.index(cell)] < 0) { dist[this.index(cell)] = 0; queue.push(cell); }
    for (let at = 0; at < queue.length; at++) {
      const cell = queue[at], d = dist[this.index(cell)];
      for (const move of PACMAN_MOVES) {
        const next = this.next(cell, move);
        if (!this.open(next.x, next.y) || dist[this.index(next)] >= 0) continue;
        dist[this.index(next)] = d + 1;
        queue.push(next);
      }
    }
    return dist;
  }

  // Where a ghost is heading: in the arcade's manner, each ghost has its own way of chasing.
  target(ghost) {
    const pac = this.pac, ahead = PACMAN_STEP[this.heading];
    if (!this.chasing) return ghost.corner;
    if (ghost.name === "Pinky") return { x: pac.x + 4 * ahead.x, y: pac.y + 4 * ahead.y };
    if (ghost.name === "Inky") {
      const blinky = this.ghosts[0], pivot = { x: pac.x + 2 * ahead.x, y: pac.y + 2 * ahead.y };
      return { x: 2 * pivot.x - blinky.x, y: 2 * pivot.y - blinky.y };
    }
    if (ghost.name === "Clyde" && (ghost.x - pac.x) ** 2 + (ghost.y - pac.y) ** 2 <= 36) return ghost.corner;
    return pac;
  }

  // A ghost out of the house goes on without turning back, at a crossing taking the way that
  // brings it nearest its target as the crow flies; a blue ghost takes one at random.
  wander(ghost, target) {
    const options = PACMAN_MOVES.filter((move) => move !== PACMAN_REVERSE[ghost.dir] && this.open(this.next(ghost, move).x, this.next(ghost, move).y));
    let dir = PACMAN_REVERSE[ghost.dir];
    if (options.length && ghost.scared && ghost.state === "out") dir = options[Math.floor(this.random() * options.length)];
    else if (options.length) {
      const far = (move) => { const cell = this.next(ghost, move); return (cell.x - target.x) ** 2 + (cell.y - target.y) ** 2; };
      dir = options.reduce((best, move) => far(move) < far(best) ? move : best);
    }
    ghost.dir = dir;
    Object.assign(ghost, this.next(ghost, dir));
  }

  // How many squares a ghost moves this step: its pace is kept as credit, a square for each whole one.
  pace(ghost) {
    if (ghost.state === "eyes") return 2;
    if (ghost.state === "home") return 1;            // a move that only looks whether its time to leave has come
    let pace = PACMAN_PACE[Math.min(this.level, PACMAN_PACE.length) - 1];
    if (ghost.name === "Blinky" && !ghost.scared && this.left <= 20) pace += 0.1;
    const tunnel = this.mark(0, ghost.y) !== "#" && (ghost.x < 4 || ghost.x >= this.cols - 4);
    if (ghost.scared || tunnel) pace /= 2;
    ghost.credit += pace;
    const squares = Math.floor(ghost.credit);
    ghost.credit -= squares;
    return squares;
  }

  moveGhost(ghost) {
    const above = { x: this.door.x, y: this.door.y - 1 }, inside = this.door.y + 1;
    if (ghost.state === "home") {
      if (this.tick >= ghost.release) ghost.state = "leaving";
      return;
    }
    if (ghost.state === "leaving") {                 // to the door's column, then up and out
      if (ghost.x !== this.door.x) ghost.x += Math.sign(this.door.x - ghost.x);
      else if (--ghost.y === above.y) { ghost.state = "out"; ghost.dir = "left"; }
      return;
    }
    if (ghost.state === "eyes") {                    // back to the house, in through the door
      if (ghost.x === above.x && ghost.y >= above.y && ghost.y < inside) ghost.y++;
      else return this.wander(ghost, above);
      if (ghost.y === inside) { ghost.state = "home"; ghost.release = this.tick; }
      return;
    }
    this.wander(ghost, this.target(ghost));
  }

  frighten() {
    this.frightened = Math.max(16, 56 - 8 * (this.level - 1));
    this.combo = 0;
    for (const ghost of this.ghosts) if (ghost.state === "out") { ghost.scared = true; ghost.dir = PACMAN_REVERSE[ghost.dir]; }
  }

  // Pac-Man meets a ghost on a square, or passes it between two: a blue one is eaten,
  // any other one catches it. True when a life was lost.
  meet(from) {
    for (const ghost of this.ghosts) {
      if (ghost.state !== "out") continue;
      const same = ghost.x === this.pac.x && ghost.y === this.pac.y;
      const crossed = from && ghost.x === from.x && ghost.y === from.y && ghost.px === this.pac.x && ghost.py === this.pac.y;
      if (!same && !crossed) continue;
      if (ghost.scared) {
        this.score += 200 << this.combo++;
        Object.assign(ghost, { state: "eyes", scared: false });
        this.event = "ghost";
        continue;
      }
      this.lives--;
      this.event = "died"; this.caughtBy = ghost.name;
      if (!this.lives) { this.alive = false; this.cause = `was caught by ${ghost.name}`; }
      else this.restart();
      return true;
    }
    return false;
  }

  // One step: Pac-Man moves (null, or a way into a wall, stays put), eats, and the ghosts move.
  step(move) {
    if (!this.alive) return;
    this.event = ""; this.steps++; this.tick++; this.hungry++;
    const from = { ...this.pac };
    if (move && this.canGo(move)) { this.heading = move; this.pac = this.next(this.pac, move); }
    const at = this.index(this.pac);
    if (this.dots[at]) {
      this.score += this.dots[at] === 2 ? 50 : 10;
      if (this.dots[at] === 2) { this.frighten(); this.event = "power"; }
      this.dots[at] = 0; this.left--; this.hungry = 0;
    }
    if (this.meet(null)) return;
    for (const ghost of this.ghosts) {
      ghost.px = ghost.x; ghost.py = ghost.y;
      for (let n = this.pace(ghost); n > 0; n--) this.moveGhost(ghost);
    }
    if (this.meet(from)) return;
    if (this.frightened && !--this.frightened) for (const ghost of this.ghosts) ghost.scared = false;
    else if (!this.frightened && ++this.modeTicks >= PACMAN_MODES[this.mode][1]) {
      this.mode++; this.modeTicks = 0;              // a ghost turns round when the mode changes
      for (const ghost of this.ghosts) if (ghost.state === "out") ghost.dir = PACMAN_REVERSE[ghost.dir];
    }
    if (!this.left) {
      this.level++; this.cleared++; this.event = "level";
      this.fill(); this.restart();
      return;
    }
    if (this.hungry > PACMAN_HUNGER) { this.alive = false; this.cause = "stopped eating"; }
  }

  // What each open way leads to: an index into PACMAN_OUTCOMES.
  outcomes() {
    const here = this.index(this.pac), dots = [];
    for (let i = 0; i < this.dots.length; i++) if (this.dots[i]) dots.push({ x: i % this.cols, y: Math.floor(i / this.cols) });
    const ghosts = this.ghosts.filter((ghost) => ghost.state === "out").map((ghost) => ({ ghost, dist: this.distances([ghost]) }));
    // The food is a blue ghost, if one is near enough to catch while it is still blue (it moves
    // every other step, at random), and otherwise the dots.
    const blue = ghosts.filter(({ ghost, dist }) => ghost.scared && dist[here] >= 0 && 2 * dist[here] <= this.frightened - 2);
    const toFood = this.distances(blue.length ? blue.map(({ ghost }) => ghost) : dots);
    return Object.fromEntries(PACMAN_MOVES.filter((move) => this.canGo(move)).map((move) => {
      const at = this.index(this.next(this.pac, move));
      const trial = this.clone();
      trial.step(move);
      if (trial.lives < this.lives) return [move, 3];
      if (ghosts.some(({ ghost, dist }) => !ghost.scared && dist[at] <= PACMAN_DANGER && dist[at] < dist[here])) return [move, 2];
      return [move, toFood[at] >= 0 && toFood[at] < toFood[here] ? 0 : 1];
    }));
  }

  // The question for this position: the options are the open ways, described.
  question() {
    const outcomes = this.outcomes();
    return { outcomes, question: { type: "choice", instructions: PACMAN_INSTRUCTIONS,
      criteria: Object.fromEntries(Object.entries(outcomes).map(([move, level]) => [PACMAN_OPTIONS[move], PACMAN_OUTCOMES[level]])) } };
  }
}

if (typeof module !== "undefined") {
  module.exports = { PACMAN_MAZE, PACMAN_MOVES, PACMAN_STEP, PACMAN_OPTIONS, PACMAN_STATE, PACMAN_INSTRUCTIONS, PACMAN_OUTCOMES,
                     PACMAN_RISKY_FROM, PACMAN_DANGER, PACMAN_LIVES, PACMAN_GHOSTS, PacmanWorld, pacmanMove };
}
