// Plays Dino Arena without a browser or a board: the game from webapp/static/games.html, driven
// by the perfect policy from games/dino_policy.py, at several decision latencies.
//
//   node tools/sim_dino.js
//
// It checks two things the web page depends on: that the game only produces states the policy
// (and so the training set) knows, and that the policy clears the course when decisions arrive
// quickly. The crash counts at higher latencies are what the page's delay slider demonstrates.
const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");
const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(root, "webapp/static/games.html"), "utf8");
const script = html.slice(html.indexOf("<script>") + 8, html.indexOf("// Arena: Laya's lane"));
const table = JSON.parse(execFileSync("python3", ["-c",
  "import json; from games import dino_policy as dp; print(json.dumps({json.dumps(s): a for s, a in dp.all_states()}))"],
  { cwd: root, encoding: "utf8" }));
const { DinoWorld, STEP } = new Function(script + "; return { DinoWorld, STEP };")();
const key = (s) => JSON.stringify(s).replaceAll(",", ", ").replaceAll(":", ": ");

const seenStates = new Set();
function run(latency, period, seconds, seed) {
  const world = new DinoWorld(seed);
  let action = "run", crashes = 0, cleared = 0, unknown = 0, pending = [], nextDecision = 0;
  for (let t = 0; t < seconds; t += STEP) {
    if (t >= nextDecision) {            // observe now, act `latency` later
      const seen = key(world.observe());
      if (!(seen in table)) { unknown++; console.log("state outside the training set:", seen); }
      pending.push({ at: t + latency, action: table[seen] || "run" });
      seenStates.add(seen);
      nextDecision = t + period;
    }
    while (pending.length && pending[0].at <= t) action = pending.shift().action;
    if (world.step(STEP, action)) action = "run";
    if (!world.alive) { crashes++; cleared += world.cleared; world.reset(seed + crashes * 17); action = "run"; pending = []; }
  }
  return { crashes, cleared: cleared + world.cleared, unknown };
}
for (const [latency, period] of [[0.025, 0.025], [0.040, 0.030], [0.080, 0.080], [0.150, 0.150], [0.300, 0.300], [0.500, 0.500]]) {
  let total = { crashes: 0, cleared: 0, unknown: 0 };
  for (let seed = 1; seed <= 5; seed++) { const r = run(latency, period, 240, seed * 101); for (const k in total) total[k] += r[k]; }
  console.log(`latency ${String(latency * 1000).padStart(3)} ms: ${total.crashes} crashes, ${total.cleared} cleared over 20 min of play, unknown states ${total.unknown}`);
}
console.log("distinct states seen:", seenStates.size, "of", Object.keys(table).length);
