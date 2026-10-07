// Rock Paper Scissors: what the model is asked. Loaded by rps.html and tools/sim_rps.js; the
// texts are repeated in games/rps_policy.py, which explains them, and must stay the same there.
//
// Here the model does read the raw thing itself: it is given the opponent's recent throws as
// plain words and asked which throw comes next. It answers with a throw that is frequent in
// that list most of the time, which is a fair guess at a person's habits. The harness then
// plays whatever beats the model's answer; that step is arithmetic, not a decision.
"use strict";

const RPS_THROWS = ["rock", "paper", "scissors"];
const RPS_BEATS = { rock: "scissors", paper: "rock", scissors: "paper" };       // a throw, and what it beats
const RPS_MEMORY = 8;                 // how many of the opponent's last throws the model is shown
const RPS_INSTRUCTIONS = "Which throw will the opponent make next?";

const rpsState = (throws) => `The opponent's last throws, oldest first: ${throws.slice(-RPS_MEMORY).join(", ")}.`;
const rpsQuestion = () => ({ type: "choice", instructions: RPS_INSTRUCTIONS, criteria: RPS_THROWS });
// The throw that beats `expected`.
const rpsCounter = (expected) => RPS_THROWS.find((mine) => RPS_BEATS[mine] === expected);
// 1 if `mine` beats `theirs`, 0 for a tie, -1 if it loses.
const rpsScore = (mine, theirs) => (mine === theirs ? 0 : RPS_BEATS[mine] === theirs ? 1 : -1);

if (typeof module !== "undefined") {
  module.exports = { RPS_THROWS, RPS_BEATS, RPS_MEMORY, RPS_INSTRUCTIONS, rpsState, rpsQuestion, rpsCounter, rpsScore };
}
