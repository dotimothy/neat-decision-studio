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
//   llm.reads = "..."                 how this game reads a move out of a reply, for the panel
//
// The panel says how the model is played with, not only what it did: what it was sent for its
// last move, word for word, what it answered, how the answer was read, and the method.
"use strict";

function createLLMPlayer(section) {
  const make = (tag, props = {}, ...children) => { const node = Object.assign(document.createElement(tag), props); node.append(...children); return node; };
  if (!document.getElementById("llm-style")) {
    document.head.append(make("style", { id: "llm-style", textContent: `
      .llm-stats { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }
      .llm-stats div { background: var(--chip); border-radius: 8px; padding: 8px 10px; }
      .llm-stats b { display: block; font-variant-numeric: tabular-nums; font-size: 16px; }
      .llm-stats span { font-size: 12px; color: var(--muted); }
      .llm-panel pre { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 240px; overflow: auto; }
      .llm-panel .llm-label { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 8px 0 3px; }
      ol.llm-method { margin: 6px 0 0; padding-left: 20px; font-size: 13.5px; color: var(--muted); display: grid; gap: 7px; }
      ol.llm-method b { color: var(--ink); font-weight: 600; }
    ` }));
  }
  const tile = (label) => { const value = make("b", { textContent: "–" }); return [make("div", {}, value, make("span", { textContent: label })), value]; };
  const model = make("p", { className: "asked" });
  const [movesTile, movesValue] = tile("moves it has made"), [timeTile, timeValue] = tile("time a move");
  const [straysTile, straysValue] = tile("replies that named no legal move"), [tokensTile, tokensValue] = tile("tokens in its last reply");
  const system = make("pre", { textContent: "–" }), user = make("pre", { textContent: "–" });
  const said = make("pre", { textContent: "–" }), read = make("p", { className: "note", textContent: "It has not moved yet." });
  const settings = make("li"), reading = make("li");
  const method = make("ol", { className: "llm-method" },
    make("li", {}, make("b", { textContent: "A general chat model. " }), "It was not trained for this game, and nothing here is fitted to it. It is whatever chat model "
      + "NEAT GenAI Studio is serving on this board, asked through its OpenAI-compatible chat endpoint; this app's server passes each question on."),
    make("li", {}, make("b", { textContent: "Asked in words, once a move. " }), "It gets two messages: one saying what game it is playing, which side it has and to answer "
      + "with only its move, and one describing the position and listing the legal moves. They are shown above as they were sent."),
    make("li", {}, make("b", { textContent: "No memory. " }), "Each move is a new conversation. What it knows of the game so far is what the message tells it."),
    settings, reading,
    make("li", {}, make("b", { textContent: "A reply that is not a move. " }), "If nothing in the reply is a legal move, a legal move is played at random in its place and the "
      + "reply is counted above, so a game is never stuck on it."),
    make("li", {}, make("b", { textContent: "No look-ahead. " }), "It searches nothing and scores nothing: it writes the move its training makes most likely after that text. "
      + "It takes its time on the MLA all the same: a reply is generated a token at a time."));
  const panel = make("div", { className: "block llm-panel", hidden: true },
    make("h2", { textContent: "The language model it plays" }), model,
    make("div", { className: "llm-stats" }, movesTile, timeTile, straysTile, tokensTile),
    make("div", { className: "block" }, make("h2", { textContent: "What it was sent for its last move" }),
      make("div", { className: "llm-label", textContent: "The standing instruction (system message)" }), system,
      make("div", { className: "llm-label", textContent: "The position (user message)" }), user),
    make("div", { className: "block" }, make("h2", { textContent: "What it answered" }), said, read),
    make("div", { className: "block" }, make("h2", { textContent: "Its method" }), method));
  section.append(panel);
  const llm = { moves: 0, strays: 0, seconds: 0, available: false, name: null, error: null,
                reads: "The first legal move named in the reply is the one played." };
  let limit = 24;
  const showMethod = () => {
    settings.replaceChildren(make("b", { textContent: "Settings. " }), `Temperature 0.2, so it nearly always gives its likeliest answer; its thinking mode is switched off; `
      + `and the reply is cut at ${limit} tokens, enough for a move and little else.`);
    reading.replaceChildren(make("b", { textContent: "Reading the reply. " }), llm.reads);
  };

  llm.status = async () => {
    try { Object.assign(llm, await (await fetch("/api/llm")).json()); } catch (_) { llm.available = false; llm.error = "the board did not answer"; }
    llm.name = (llm.models || [])[0] || null;
    model.textContent = llm.available ? `${llm.name}, served by NEAT GenAI Studio at ${llm.url}`
      : `No language model to play: ${llm.error}. Start NEAT GenAI Studio on the board (neat-ai) and load a chat model in it.`;
    return llm;
  };
  llm.show = (on) => { panel.hidden = !on; if (on) { showMethod(); llm.status(); } };
  llm.ask = async (systemText, userText, maxTokens = 24) => {
    limit = maxTokens;
    system.textContent = systemText; user.textContent = userText;
    showMethod();
    const response = await fetch("/api/llm/chat", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ max_tokens: maxTokens, messages: [{ role: "system", content: systemText }, { role: "user", content: userText }] }) });
    const body = await response.json();
    if (body.error) { llm.available = false; throw new Error(body.error); }
    return body;
  };
  // What it said, what was played for it, and whether its reply was a legal move at all.
  llm.played = (reply, move, legal) => {
    llm.moves++; llm.seconds += reply.ms / 1000;
    if (!legal) llm.strays++;
    said.textContent = reply.text || "(nothing)";
    read.textContent = (legal ? `Read as ${move}, which is legal, and played.` : `No legal move could be read out of it, so ${move} was played at random.`)
      + ` It took ${reply.ms < 1000 ? Math.round(reply.ms) + " ms" : (reply.ms / 1000).toFixed(1) + " s"}` + (reply.model ? ` (${reply.model}).` : ".");
    const each = llm.seconds / llm.moves;
    movesValue.textContent = llm.moves.toLocaleString();
    timeValue.textContent = each < 1 ? Math.round(each * 1000) + " ms" : each.toFixed(1) + " s";
    straysValue.textContent = `${llm.strays} of ${llm.moves}`;
    tokensValue.textContent = reply.tokens ?? "–";
  };
  llm.reset = () => {
    llm.moves = llm.strays = llm.seconds = 0;
    for (const value of [movesValue, timeValue, straysValue, tokensValue]) value.textContent = "–";
    system.textContent = user.textContent = said.textContent = "–"; read.textContent = "It has not moved yet.";
  };
  return llm;
}
