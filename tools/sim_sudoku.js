// Plays Sudoku without a browser or a board, with the puzzle code the page uses
// (webapp/static/sudoku-core.js), taking a best-described cell and then a best-described digit
// every time (the first of them when several are equally good). tools/sudoku_check.py confirms
// on the board that the model picks a best-described option.
//
//   node tools/sim_sudoku.js [puzzles]
//
// It shows what the descriptions alone are worth at each level, and checks the puzzles: one
// solution each, and of the level they claim.
const path = require("path");
const g = require(path.join(__dirname, "..", "webapp/static/sudoku-core.js"));

const puzzles = Number(process.argv[2] || 200);
const first = (outcomes) => { const best = Math.min(...Object.values(outcomes)); return Object.keys(outcomes).find((key) => outcomes[key] === best); };
for (const level of g.SUDOKU_LEVELS) {
  let blanks = 0, mistakes = 0, risks = 0, clean = 0, unique = 0, ofLevel = 0, tokens = 0, ms = 0;
  for (let seed = 1; seed <= puzzles; seed++) {
    const started = process.hrtime.bigint();
    const game = new g.SudokuGame(seed * 104729, level);
    ms += Number(process.hrtime.bigint() - started) / 1e6;
    const puzzle = Uint8Array.from(game.grid);
    unique += g.sudokuCount(Uint8Array.from(puzzle), 2) === 1;
    const naked = g.sudokuSolvesByCertain(puzzle, false), both = g.sudokuSolvesByCertain(puzzle, true);
    ofLevel += level === "easy" ? naked : level === "medium" ? both && !naked : !both;
    while (!game.done) {
      const { cells, outcomes } = game.cellQuestion();
      const cell = cells.find((i) => g.sudokuCellName(i) === first(outcomes));
      const asked = game.digitQuestion(cell);
      // The longest question, in words, as a stand-in for its length in tokens.
      tokens = Math.max(tokens, JSON.stringify(asked.question.criteria).split(/\W+/).length);
      game.place(cell, first(asked.outcomes));
    }
    blanks += game.blanks; mistakes += game.mistakes; risks += game.risks; clean += game.mistakes === 0;
  }
  console.log(`${level}: ${puzzles} puzzles, ${unique} with one solution, ${ofLevel} of that level, ${(blanks / puzzles).toFixed(1)} cells to fill, `
    + `${(ms / puzzles).toFixed(0)} ms to make one`);
  console.log(`  ${(risks / puzzles).toFixed(2)} risks and ${(mistakes / puzzles).toFixed(2)} mistakes per puzzle, ${clean} puzzles without a mistake, `
    + `${(2 * blanks / puzzles).toFixed(0)} decisions per puzzle`);
}
