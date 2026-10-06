// Sudoku: the puzzle, and how the harness describes it to the model. Loaded by sudoku.html and
// by tools/sim_sudoku.js; the texts are repeated in games/sudoku_policy.py, which explains the
// wording, and must stay the same there.
//
// The model is not shown the grid: given the digits a cell sees, it does not find the one that
// is missing (see games/sudoku_policy.py). So, as in Snake, the harness does the looking and
// the options of a `choice` question say what it found. Filling a cell takes two decisions:
//
//   which cell   five empty cells are offered, each described by how sure its digit is
//   which digit  the nine digits, each described by what it would do in that cell
//
// A digit is certain when it is the only one the cell's row, column and box leave over, or
// when the cell is the only place left for it in one of them. Easy puzzles need only the
// first, medium ones both, and hard ones run out of certain cells, and then the model has to
// take a risk. A wrong digit is counted and replaced by the right one, so a puzzle always
// gets finished and what is counted is the mistakes.
"use strict";

const SUDOKU_LEVELS = ["easy", "medium", "hard"];
const SUDOKU_OFFERED = 5;            // cells offered per "which cell" question
const SUDOKU_BLANKS = { easy: 46 };   // an easy puzzle also keeps more of its digits
const SUDOKU_CELL_STATE = "The puzzle needs a cell that is safe and certain.";
const SUDOKU_CELL_INSTRUCTIONS = "Which cell should be filled next?";
// How sure a cell's digit is, best first. The model reads these exact words.
const SUDOKU_CELL_OUTCOMES = ["safe and certain, one digit must go here", "a small risk, two digits fit", "a big risk, many digits fit"];
const SUDOKU_DIGIT_STATE = "The cell needs a digit that is safe and certain.";
const SUDOKU_DIGIT_INSTRUCTIONS = "Which digit should go in the cell?";
// What a digit would do in the cell, best first.
const SUDOKU_DIGIT_OUTCOMES = ["safe and certain", "safe but a guess", "repeats in the row, breaks the puzzle",
                               "repeats in the column, breaks the puzzle", "repeats in the box, breaks the puzzle"];
const SUDOKU_BREAKS_FROM = 2;        // digit outcomes from this index on break the rules

const SUDOKU_ROW = [], SUDOKU_COLUMN = [], SUDOKU_BOX = [];       // per cell: the cells of its row, column and box
for (let i = 0; i < 81; i++) {
  const row = Math.floor(i / 9), column = i % 9, box = 27 * Math.floor(row / 3) + 3 * Math.floor(column / 3);
  SUDOKU_ROW.push(Array.from({ length: 9 }, (_, k) => row * 9 + k));
  SUDOKU_COLUMN.push(Array.from({ length: 9 }, (_, k) => k * 9 + column));
  SUDOKU_BOX.push(Array.from({ length: 9 }, (_, k) => box + 9 * Math.floor(k / 3) + k % 3));
}
const sudokuCellName = (i) => `r${Math.floor(i / 9) + 1}c${i % 9 + 1}`;
const sudokuBits = (mask) => { let n = 0; for (; mask; mask &= mask - 1) n++; return n; };

function sudokuRandom(seed) {
  return () => {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function sudokuShuffle(items, random) {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
  return items;
}

// Which unit rules a digit out of a cell: 0 none, else 2 row, 3 column, 4 box (the digit outcomes).
function sudokuClash(grid, i, digit) {
  if (SUDOKU_ROW[i].some((j) => grid[j] === digit)) return 2;
  if (SUDOKU_COLUMN[i].some((j) => grid[j] === digit)) return 3;
  if (SUDOKU_BOX[i].some((j) => grid[j] === digit)) return 4;
  return 0;
}
// The digits a cell's row, column and box leave over, as a bit mask (bit d for digit d).
function sudokuCandidates(grid, i) {
  let used = 0;
  for (const unit of [SUDOKU_ROW[i], SUDOKU_COLUMN[i], SUDOKU_BOX[i]]) for (const j of unit) used |= 1 << grid[j];
  return 0x3fe & ~used;
}
// The digit that must go in an empty cell, or 0: the only one left for the cell, or (with
// `hidden`) one that has no other place left in the cell's row, column or box.
function sudokuCertain(grid, i, hidden = true) {
  const mask = sudokuCandidates(grid, i);
  if (sudokuBits(mask) === 1) return 31 - Math.clz32(mask);
  if (!hidden) return 0;
  for (let digit = 1; digit <= 9; digit++) {
    if (!(mask & 1 << digit)) continue;
    for (const unit of [SUDOKU_ROW[i], SUDOKU_COLUMN[i], SUDOKU_BOX[i]]) {
      if (unit.every((j) => j === i || grid[j] || !(sudokuCandidates(grid, j) & 1 << digit))) return digit;
    }
  }
  return 0;
}
// Whether writing certain digits alone finishes the puzzle.
function sudokuSolvesByCertain(puzzle, hidden) {
  const grid = Uint8Array.from(puzzle);
  for (let progress = true; progress;) {
    progress = false;
    for (let i = 0; i < 81; i++) {
      if (grid[i]) continue;
      const digit = sudokuCertain(grid, i, hidden);
      if (digit) { grid[i] = digit; progress = true; }
    }
  }
  return grid.every((digit) => digit);
}
// How many ways the puzzle can be finished, counting no further than `limit`.
function sudokuCount(grid, limit) {
  let at = -1, fewest = 10;
  for (let i = 0; i < 81; i++) {
    if (grid[i]) continue;
    const n = sudokuBits(sudokuCandidates(grid, i));
    if (n < fewest) { fewest = n; at = i; }
  }
  if (at < 0) return 1;
  let count = 0;
  for (let digit = 1, mask = sudokuCandidates(grid, at); digit <= 9 && count < limit; digit++) {
    if (!(mask & 1 << digit)) continue;
    grid[at] = digit;
    count += sudokuCount(grid, limit - count);
  }
  grid[at] = 0;
  return count;
}

// A finished grid, then cells are taken away for as long as the puzzle stays of its level.
function sudokuGenerate(seed, level) {
  for (let attempt = 0; ; attempt++) {
    const random = sudokuRandom(seed + attempt * 7919);
    const solution = new Uint8Array(81);
    (function fill(i) {
      if (i === 81) return true;
      const mask = sudokuCandidates(solution, i);
      for (const digit of sudokuShuffle([1, 2, 3, 4, 5, 6, 7, 8, 9], random)) {
        if (!(mask & 1 << digit)) continue;
        solution[i] = digit;
        if (fill(i + 1)) return true;
      }
      solution[i] = 0;
      return false;
    })(0);
    const puzzle = Uint8Array.from(solution);
    let blanks = 0;
    for (const i of sudokuShuffle(Array.from({ length: 81 }, (_, k) => k), random)) {
      if (blanks === (SUDOKU_BLANKS[level] || 81)) break;
      puzzle[i] = 0;
      const keeps = level === "easy" ? sudokuSolvesByCertain(puzzle, false)
        : level === "medium" ? sudokuSolvesByCertain(puzzle, true) : sudokuCount(Uint8Array.from(puzzle), 2) === 1;
      if (keeps) blanks++; else puzzle[i] = solution[i];
    }
    // A medium puzzle has to need the second kind of certain digit, a hard one more than both.
    const needsMore = level === "easy" || !sudokuSolvesByCertain(puzzle, level === "hard");
    if (needsMore || attempt === 20) return { puzzle, solution };
  }
}

class SudokuGame {
  constructor(seed, level = "easy") { this.reset(seed, level); }

  reset(seed, level) {
    if (SUDOKU_LEVELS.includes(level)) this.level = level;
    const { puzzle, solution } = sudokuGenerate(seed, this.level);
    this.solution = solution;
    this.grid = Uint8Array.from(puzzle);
    this.given = puzzle.map((digit) => digit ? 1 : 0);
    this.wrong = new Uint8Array(81);         // per cell: the wrong digit the model wrote there, if it did
    this.random = sudokuRandom(seed ^ 0x5bd1e995);
    this.blanks = puzzle.reduce((count, digit) => count + !digit, 0);
    this.filled = 0; this.mistakes = 0; this.risks = 0;
  }

  get done() { return this.filled === this.blanks; }
  empties() { const cells = []; for (let i = 0; i < 81; i++) if (!this.grid[i]) cells.push(i); return cells; }

  // How sure a cell's digit is: an index into SUDOKU_CELL_OUTCOMES.
  cellOutcome(i) {
    if (sudokuCertain(this.grid, i)) return 0;
    return sudokuBits(sudokuCandidates(this.grid, i)) === 2 ? 1 : 2;
  }
  // What each digit would do in a cell: indexes into SUDOKU_DIGIT_OUTCOMES, for digits 1 to 9.
  digitOutcomes(i) {
    const certain = sudokuCertain(this.grid, i);
    return Array.from({ length: 9 }, (_, k) => k + 1 === certain ? 0 : sudokuClash(this.grid, i, k + 1) || 1);
  }

  // The cells to choose from: one of the surest on the board, and others taken at random.
  offer() {
    const cells = this.empties();
    const outcomes = cells.map((i) => this.cellOutcome(i)), best = Math.min(...outcomes);
    const surest = cells.filter((_, k) => outcomes[k] === best);
    const first = surest[Math.floor(this.random() * surest.length)];
    const others = sudokuShuffle(cells.filter((i) => i !== first), this.random).slice(0, SUDOKU_OFFERED - 1);
    return [first, ...others].sort((a, b) => a - b);
  }

  cellQuestion() {
    const cells = this.offer(), outcomes = Object.fromEntries(cells.map((i) => [sudokuCellName(i), this.cellOutcome(i)]));
    return { cells, outcomes, question: { type: "choice", instructions: SUDOKU_CELL_INSTRUCTIONS,
      criteria: Object.fromEntries(cells.map((i) => [sudokuCellName(i), SUDOKU_CELL_OUTCOMES[outcomes[sudokuCellName(i)]]])) } };
  }
  digitQuestion(i) {
    const levels = this.digitOutcomes(i), outcomes = Object.fromEntries(levels.map((level, k) => [String(k + 1), level]));
    return { outcomes, question: { type: "choice", instructions: SUDOKU_DIGIT_INSTRUCTIONS,
      criteria: Object.fromEntries(levels.map((level, k) => [String(k + 1), SUDOKU_DIGIT_OUTCOMES[level]])) } };
  }

  // Writes the model's digit. A wrong one is counted and replaced by the right one.
  place(i, digit) {
    const right = Number(digit) === this.solution[i];
    if (!sudokuCertain(this.grid, i)) this.risks++;
    if (!right) { this.wrong[i] = Number(digit); this.mistakes++; }
    this.grid[i] = this.solution[i];
    this.filled++;
    return right;
  }
}

if (typeof module !== "undefined") {
  module.exports = { SUDOKU_LEVELS, SUDOKU_OFFERED, SUDOKU_CELL_STATE, SUDOKU_CELL_INSTRUCTIONS, SUDOKU_CELL_OUTCOMES,
                     SUDOKU_DIGIT_STATE, SUDOKU_DIGIT_INSTRUCTIONS, SUDOKU_DIGIT_OUTCOMES, SUDOKU_BREAKS_FROM,
                     sudokuCellName, sudokuCandidates, sudokuCertain, sudokuSolvesByCertain, sudokuCount, sudokuGenerate, SudokuGame };
}
