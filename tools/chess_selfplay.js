// Two copies of the chess model play each other on the DevKit, from the host: the same
// questions and the same way of choosing a move as the page (webapp/static/chess-core.js),
// sent to a `laya serve` of its own over ssh, so the app's loaded model is left alone.
// It is how the repetition rule is checked with the real model.
//
//   node tools/chess_selfplay.js [games] [plies] [rule|norule]
//
// `norule` chooses by rating alone, as before the rule: a draw by repetition counts as a draw.
const path = require("path"), { spawn } = require("child_process"), readline = require("readline");
const { Chess } = require(path.join(__dirname, "..", "webapp/static/vendor/chess.js"));
const core = require(path.join(__dirname, "..", "webapp/static/chess-core.js"));

const games = Number(process.argv[2] || 2), plies = Number(process.argv[3] || 160), rule = process.argv[4] !== "norule";
const board = process.env.LAYA_BOARD || "sima@192.168.91.225", remote = process.env.LAYA_REMOTE_DIR || "/media/nvme/laya";
const child = spawn("ssh", ["-o", "BatchMode=yes", board, `${remote}/laya serve ${remote}/model-chess`], { stdio: ["pipe", "pipe", "ignore"] });
const waiting = [];
readline.createInterface({ input: child.stdout }).on("line", (line) => {
  if (line.startsWith('{"answers"') || line.startsWith('{"error"')) waiting.shift()(JSON.parse(line));
});
const ask = (payload) => new Promise((resolve) => { waiting.push(resolve); child.stdin.write(JSON.stringify(payload) + "\n"); });

(async () => {
  const endings = {}; let decisions = 0, ms = 0, back = 0, total = 0;
  for (let n = 0; n < games; n++) {
    const game = new Chess();
    while (!game.game_over() && game.history().length < plies) {
      const moves = game.moves({ verbose: true }), state = core.chessState(game, moves), seen = core.chessSeen(game);
      const rated = moves.map((move) => ({ move, exact: core.chessExact(game, move), seen: core.chessAfter(game, move, seen), win: null }));
      const playable = rule ? core.chessCandidates(rated) : rated;
      for (const item of playable) {
        if (item.exact !== null) { item.win = item.exact; continue; }
        if (!rule && item.seen >= 2) { item.win = 0.5; continue; }          // before the rule: a third time is a draw
        const answer = await ask({ state, questions: { q: core.chessQuestion(game, item.move) } });
        if (answer.error) throw new Error(answer.error);
        item.win = core.chessWinChance(answer.answers.q); decisions++; ms += answer.usage.mla_ms;
      }
      const worth = rule ? core.chessWorth : (item) => item.win;
      let top = Math.max(...playable.map(worth));
      let choices = playable.filter((item) => worth(item) === top);
      if (game.history().length < 8) choices = playable.filter((item) => worth(item) >= top - 0.02);      // the page's variety in the opening
      const chosen = choices[Math.floor(Math.random() * choices.length)];
      back += chosen.seen > 0;
      game.move(chosen.move);
    }
    const result = core.chessResult(game), how = result ? result.how : `still going after ${plies} plies`;
    endings[how] = (endings[how] || 0) + 1; total += game.history().length;
    const most = Math.max(...core.chessSeen(game).values());
    console.log(`game ${n + 1}: ${how}, ${game.history().length} plies, a position stood at most ${most} time${most === 1 ? "" : "s"}; last moves: ${game.history().slice(-8).join(" ")}`);
  }
  console.log(`${rule ? "with the rule" : "by rating alone"}: ${games} games, ended by ${JSON.stringify(endings)}; ${(back / games).toFixed(1)} moves a game back into `
    + `an earlier position; ${decisions} decisions at ${(ms / decisions).toFixed(1)} ms on the MLA`);
  child.stdin.end();
})().catch((error) => { console.error(String(error)); child.kill(); process.exit(1); });
