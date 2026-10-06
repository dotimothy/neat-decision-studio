// Snake: the game, and how the harness describes a position to the model. Loaded by snake.html
// and by tools/sim_snake.js; the texts are repeated in games/snake_policy.py, which explains
// the wording, and must stay the same there.
//
// The model is not shown the board. For each of the three moves the snake can make, the
// harness works out what the move leads to (geometry: is the square free, does it get closer
// to the food, is there room beyond it) and says so in a few words. Those descriptions are the
// options of a `choice` question, and the model picks one.
//
// How much of the board goes into those descriptions is `sight`: the snake always knows the
// board's edges and where the food is, but of its own body only what lies within `sight`
// squares of its head, and it takes the rest of the board to be empty. With the whole board
// in sight the descriptions are true; with less, a move can be called safe that is not.
"use strict";

const SNAKE_GRID = { cols: 20, rows: 14 };          // the board a world gets unless told otherwise
const SNAKE_GRID_LIMITS = { min: 6, maxCols: 60, maxRows: 40 };
const SNAKE_MOVES = ["left", "straight", "right"];
const SNAKE_STATE = "The snake needs a safe move that brings it closer to the food.";
const SNAKE_INSTRUCTIONS = "Which way should the snake go?";
// What a move can lead to, best first. The model reads these exact words.
const SNAKE_OUTCOMES = ["safe and toward the food", "safe but away from the food", "a dead end, the snake dies", "blocked, the snake dies"];
const SNAKE_FATAL_FROM = 2;          // outcomes from this index on lose the game

function snakeRandom(seed) {
  return () => {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

class SnakeWorld {
  constructor(seed, grid = SNAKE_GRID, sight = Infinity) { this.sight = sight; this.reset(seed, grid); }

  // How far the snake sees, in squares around its head; Infinity is the whole board.
  get sight() { return this.range; }
  set sight(value) { this.range = value === Infinity ? Infinity : Math.max(1, Math.round(Number(value)) || 1); }
  sees(cell) { return Math.max(Math.abs(cell.x - this.head.x), Math.abs(cell.y - this.head.y)) <= this.range; }

  // A new game, on a board of another size if `grid` is given. The descriptions the model
  // reads do not mention the board, so its size changes the game and not the question.
  reset(seed, grid) {
    if (grid) {
      const size = (value, max) => Math.max(SNAKE_GRID_LIMITS.min, Math.min(max, Math.round(Number(value)) || SNAKE_GRID_LIMITS.min));
      this.cols = size(grid.cols, SNAKE_GRID_LIMITS.maxCols);
      this.rows = size(grid.rows, SNAKE_GRID_LIMITS.maxRows);
    }
    this.random = snakeRandom(seed);
    const x = Math.min(5, this.cols >> 1), y = this.rows >> 1;
    this.body = [{ x, y }, { x: x - 1, y }, { x: x - 2, y }];      // head first
    this.heading = { x: 1, y: 0 };
    this.alive = true; this.cause = ""; this.eaten = 0; this.steps = 0; this.hungry = 0;
    this.placeFood();
  }

  get head() { return this.body[0]; }
  index(cell) { return cell.y * this.cols + cell.x; }
  inside(cell) { return cell.x >= 0 && cell.y >= 0 && cell.x < this.cols && cell.y < this.rows; }

  // Squares the snake cannot enter on its next move. The tail square is free: the tail moves on.
  // With `seenOnly`, the ones it knows of: those within its sight.
  walls(seenOnly = false) {
    const walls = new Uint8Array(this.cols * this.rows);
    for (let i = 0; i < this.body.length - 1; i++) if (!seenOnly || this.sees(this.body[i])) walls[this.index(this.body[i])] = 1;
    return walls;
  }

  placeFood() {
    const taken = new Set(this.body.map((cell) => this.index(cell))), free = [];
    for (let i = 0; i < this.cols * this.rows; i++) if (!taken.has(i)) free.push(i);
    if (!free.length) { this.food = null; return; }
    const pick = free[Math.floor(this.random() * free.length)];
    this.food = { x: pick % this.cols, y: Math.floor(pick / this.cols) };
  }

  // With y growing downward, a left turn takes (dx, dy) to (dy, -dx).
  direction(move) {
    const { x, y } = this.heading;
    return move === "left" ? { x: y, y: -x } : move === "right" ? { x: -y, y: x } : { x, y };
  }
  target(move) { const d = this.direction(move); return { x: this.head.x + d.x, y: this.head.y + d.y }; }

  // Steps to every square from `start`, through free squares (-1 where there is no way).
  distances(start, walls) {
    const dist = new Int16Array(walls.length).fill(-1), queue = [start];
    dist[this.index(start)] = 0;
    for (let at = 0; at < queue.length; at++) {
      const cell = queue[at], d = dist[this.index(cell)];
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const next = { x: cell.x + dx, y: cell.y + dy };
        if (!this.inside(next) || walls[this.index(next)] || dist[this.index(next)] >= 0) continue;
        dist[this.index(next)] = d + 1;
        queue.push(next);
      }
    }
    return dist;
  }

  // What each move leads to as far as the snake can see: an index into SNAKE_OUTCOMES.
  outcomes() {
    const walls = this.walls(true);
    const toFood = this.food ? this.distances(this.food, walls) : null;
    // The head square is itself a wall in that search. The head is one step from any free neighbour, so its distance is the best neighbour's + 1.
    let headDistance = -1;
    if (toFood) for (const move of SNAKE_MOVES) {
      const cell = this.target(move);
      if (this.inside(cell) && !walls[this.index(cell)] && toFood[this.index(cell)] >= 0) {
        const d = toFood[this.index(cell)] + 1;
        if (headDistance < 0 || d < headDistance) headDistance = d;
      }
    }
    return Object.fromEntries(SNAKE_MOVES.map((move) => {
      const cell = this.target(move);
      if (!this.inside(cell) || walls[this.index(cell)]) return [move, 3];
      // Room beyond the square: fewer reachable squares than the snake is long is a dead end.
      const room = this.distances(cell, walls).reduce((count, d) => count + (d >= 0), 0);
      if (room < this.body.length) return [move, 2];
      const d = toFood ? toFood[this.index(cell)] : -1;
      return [move, d >= 0 && d < headDistance ? 0 : 1];
    }));
  }

  // The question for this position: the options are the three moves, described.
  question() {
    const outcomes = this.outcomes();
    return { outcomes, question: { type: "choice", instructions: SNAKE_INSTRUCTIONS,
      criteria: Object.fromEntries(SNAKE_MOVES.map((move) => [move, SNAKE_OUTCOMES[outcomes[move]]])) } };
  }

  step(move) {
    if (!this.alive) return;
    const cell = this.target(move), walls = this.walls();
    this.heading = this.direction(move);
    this.steps++;
    if (!this.inside(cell)) { this.alive = false; this.cause = "hit the wall"; return; }
    if (walls[this.index(cell)]) { this.alive = false; this.cause = "ran into itself"; return; }
    this.body.unshift(cell);
    if (this.food && cell.x === this.food.x && cell.y === this.food.y) {
      this.eaten++; this.hungry = 0;
      this.placeFood();
      if (!this.food) { this.alive = false; this.cause = "filled the board"; }
    } else {
      this.body.pop();
      // A snake that circles without eating has stopped playing.
      if (++this.hungry > this.cols * this.rows * 2) { this.alive = false; this.cause = "starved"; }
    }
  }
}

if (typeof module !== "undefined") {
  module.exports = { SNAKE_GRID, SNAKE_GRID_LIMITS, SNAKE_MOVES, SNAKE_STATE, SNAKE_INSTRUCTIONS, SNAKE_OUTCOMES, SNAKE_FATAL_FROM, SnakeWorld };
}
