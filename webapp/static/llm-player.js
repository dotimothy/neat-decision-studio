// A language model as an opponent: the games that can be played against one (Tic-Tac-Toe, Rock
// Paper Scissors, Chess) ask it for its move in words and read the move out of its reply.
//
// The model is whatever an OpenAI-compatible server is serving, by default NEAT GenAI Studio
// on this board (`neat-ai`, with a chat model loaded). The app's server passes the question on
// (/api/llm/chat). A reply that names no legal move is counted, and a legal move is made at
// random in its place, so a game is never stuck on it.
//
//   const llm = createLLMPlayer(sectionToAddThePanelTo);
//   const { text, ms } = await llm.ask(system, user);
//   llm.played(text, move, legal)     show what it said, and count a reply that was not a move
//   llm.status()                      { available, models, error }, asked of the server
"use strict";

function createLLMPlayer(section) {
  const make = (tag, props = {}, ...children) => { const node = Object.assign(document.createElement(tag), props); node.append(...children); return node; };
  const model = make("p", { className: "asked" }), said = make("pre", { textContent: "–" }), tally = make("p", { className: "note" });
  const panel = make("div", { className: "block", hidden: true }, make("h2", { textContent: "The language model it plays" }), model, said, tally);
  section.append(panel);
  const llm = { moves: 0, strays: 0, seconds: 0, available: false, name: null, error: null };

  llm.status = async () => {
    try { Object.assign(llm, await (await fetch("/api/llm")).json()); } catch (_) { llm.available = false; llm.error = "the board did not answer"; }
    llm.name = (llm.models || [])[0] || null;
    model.textContent = llm.available ? `${llm.name}, served by NEAT GenAI Studio at ${llm.url}`
      : `No language model to play: ${llm.error}. Start NEAT GenAI Studio on the board (neat-ai) and load a chat model in it.`;
    return llm;
  };
  llm.show = (on) => { panel.hidden = !on; if (on) llm.status(); };
  llm.ask = async (system, user, maxTokens = 24) => {
    const response = await fetch("/api/llm/chat", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ max_tokens: maxTokens, messages: [{ role: "system", content: system }, { role: "user", content: user }] }) });
    const body = await response.json();
    if (body.error) { llm.available = false; throw new Error(body.error); }
    return body;
  };
  // What it said, what was played for it, and whether its reply was a legal move at all.
  llm.played = (reply, move, legal) => {
    llm.moves++; llm.seconds += reply.ms / 1000;
    if (!legal) llm.strays++;
    said.textContent = `"${reply.text || "(nothing)"}"\n${legal ? "played: " + move : "not a legal move; played at random: " + move}`;
    const each = llm.seconds / llm.moves;
    tally.textContent = `${llm.moves} moves, ${each < 1 ? Math.round(each * 1000) + " ms" : each.toFixed(1) + " s"} each`
      + (llm.strays ? `; ${llm.strays} of its replies named no legal move` : "");
  };
  llm.reset = () => { llm.moves = llm.strays = llm.seconds = 0; said.textContent = "–"; tally.textContent = ""; };
  return llm;
}
