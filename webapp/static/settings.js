// Settings: the model manager, on every page, behind the gear in the header.
//
// It is laid out like the settings of SiMa.ai's NEAT GenAI Studio: a window over the page with
// its sections down the left. Models is what is on this board (load, unload, delete, the MLA's
// memory, Reset Accelerator); Add Model is what Hugging Face has (download); About says which
// version is running. Several models can be on the MLA at once, as many as fit in its memory.
//
//   openSettings()            open it on Models
//   openSettings("add")       on another section: "models", "add" or "about"
//
// /models, or #settings on any page, opens it as the page loads. While it is open the board
// pushes what changes (GET /api/events), so a model is seen arriving graph by graph.
"use strict";
(() => {
  const el = (tag, props = {}, ...children) => {
    const node = Object.assign(document.createElement(tag), props);
    for (const child of children) node.append(child);
    return node;
  };
  const mb = (bytes) => bytes >= 1 << 30 ? (bytes / (1 << 30)).toFixed(2) + " GB" : Math.round(bytes / (1 << 20)) + " MB";
  const left = (s) => s >= 90 ? `about ${Math.round(s / 60)} min left` : `about ${Math.max(1, Math.round(s))} s left`;

  // What is known about a checkpoint without asking Hugging Face, for models compiled locally
  // or when the board is offline: title, description, encoder, languages.
  const CHECKPOINTS = {
    "laya": ["Laya", "The general question model.", "ModernBERT-large, 421M parameters", "English"],
    "laya-typed-decisions": ["Laya typed-decisions", "Fine-tuned by upstream on typed-decision workflows.", "ModernBERT-large, 421M parameters", "English"],
    "laya-multilingual": ["Laya multilingual", "More than 100 languages, on a smaller encoder that answers about twice as fast.", "mmBERT-base, 322M parameters", "100+ languages"],
    "laya-dino": ["Laya-dino", "Decision head fine-tuned for Dino Arena.", "ModernBERT-large, 421M parameters", "English (game states)"],
    "CLM-v0.1-8B": ["CLM v0.1 8B", "A Contrastive Language Model: a frozen Qwen3-8B reads the state and each option, two small heads compare them. The same three kinds of question as Laya, on an encoder twenty times the size.", "Qwen3-8B, 8.2B parameters, in a chain of graphs", "English"],
    "d1-omni-600M": ["d1 Omni 600M", "LiquidAI's small decision model, which also reads pictures: an LFM2 encoder read in both directions, a decision head and a SigLIP2 vision tower. The same three kinds of question, about text or about a picture. An early research release.", "LFM2.5-Encoder-350M and SigLIP2, 475M parameters", "English"],
    "d1-3B": ["d1 3B", "LiquidAI's larger decision model: the LFM2.5-VL-3B language model, asked a question as a chat turn and read at the one token where its answer would begin. Text and pictures; the more careful reader of the two d1 models.", "LFM2.5-VL-3B with a SigLIP2 vision tower, 3.1B parameters, in a chain of graphs", "English"],
    "laya-chess": ["Laya-chess", "LayaChess: Laya fine-tuned on two million Stockfish-rated moves, to rate a chess move's win chance.", "ModernBERT-large, 421M parameters", "English (chess positions)"],
  };
  const PRECISIONS = { "BF16": "bfloat16 weights and activations", "A_BF16_W_INT8": "int8 weights, bfloat16 activations",
                       "A_BF16_W_INT4": "int4 weights, bfloat16 activations" };
  // One colour a loaded model, in the memory bar and beside its name: the Neat spectrum.
  const COLOURS = ["#16c8a6", "#3a86ec", "#a9c81c", "#f2801d", "#4bb54a", "#b06be0"];

  document.head.append(el("style", { textContent: `
    .settings { position: fixed; inset: 0; z-index: 200; display: none; align-items: center; justify-content: center; padding: 20px;
                background: rgb(4 8 12 / .55); backdrop-filter: blur(3px); font: 14.5px/1.45 Inter, system-ui, sans-serif; color: var(--ink); }
    .settings.open { display: flex; }
    .settings * { box-sizing: border-box; }
    .settings-card { display: flex; flex-direction: column; width: min(940px, 100%); height: min(760px, 100%); background: var(--panel);
                     border: 1px solid var(--line); border-radius: 18px; box-shadow: var(--shadow-md); overflow: hidden; }
    .settings-head { display: flex; align-items: center; gap: 10px; padding: 14px 18px; border-bottom: 1px solid var(--line); }
    .settings-head b { font-size: 18px; letter-spacing: -.01em; margin-right: auto; }
    .settings-body { display: grid; grid-template-columns: 196px minmax(0, 1fr); min-height: 0; flex: 1; }
    .settings-tabs { display: flex; flex-direction: column; gap: 4px; padding: 12px; border-right: 1px solid var(--line); }
    .settings-tabs button { all: unset; box-sizing: border-box; cursor: pointer; padding: 9px 13px; border-radius: 10px; color: var(--muted);
                            border: 1px solid transparent; font-size: 14.5px; }
    .settings-tabs button:hover { color: var(--ink); background: var(--chip); }
    .settings-tabs button.here { color: var(--accent-strong); font-weight: 650; background: color-mix(in srgb, var(--accent) 13%, var(--panel));
                                 border-color: color-mix(in srgb, var(--accent) 45%, var(--line)); }
    .settings-tabs button:focus-visible { outline: 2px solid var(--accent); }
    .settings-panel { overflow: auto; padding: 18px 20px 24px; display: none; }
    .settings-panel.here { display: block; }
    @media (max-width: 700px) {
      .settings { padding: 0; } .settings-card { border-radius: 0; height: 100%; }
      .settings-body { grid-template-columns: minmax(0, 1fr); grid-template-rows: auto minmax(0, 1fr); }
      .settings-tabs { flex-direction: row; border-right: 0; border-bottom: 1px solid var(--line); overflow: auto; }
    }
    .settings .sbtn { font: inherit; color: var(--ink); background: var(--chip); border: 1px solid var(--line); border-radius: 10px;
                      padding: 6px 14px; cursor: pointer; min-width: 0; white-space: nowrap; transition: .16s ease; }
    .settings .sbtn:hover:not(:disabled) { color: var(--accent); border-color: color-mix(in srgb, var(--accent) 55%, var(--line)); }
    .settings .sbtn:disabled { opacity: .5; cursor: default; }
    .settings .sbtn.primary { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); font-weight: 600; }
    .settings .sbtn.primary:hover:not(:disabled) { background: var(--accent-strong); border-color: var(--accent-strong); color: var(--accent-ink); }
    .settings .sbtn.danger:hover:not(:disabled) { border-color: var(--bad); color: var(--bad); }
    .settings .icon { width: 34px; height: 34px; padding: 0; display: inline-flex; align-items: center; justify-content: center; border-radius: 10px; }
    .settings .label { font-size: 12px; text-transform: uppercase; letter-spacing: .07em; color: var(--muted); font-weight: 650; }
    .settings .note { color: var(--muted); font-size: 13px; margin: 6px 0 0; }
    .settings .note a, .settings .facts a { color: var(--accent); }
    .settings .block { padding: 0 0 16px; margin: 0 0 16px; border-bottom: 1px solid var(--line); }
    .settings .row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; }
    .settings .row .grow { margin-right: auto; }
    .settings .said { color: var(--bad); font-size: 13.5px; margin: 8px 0 0; overflow-wrap: anywhere; }
    .settings .meter { display: flex; height: 10px; margin-top: 8px; background: var(--track); border-radius: 999px; overflow: hidden; }
    .settings .meter i { flex: none; height: 100%; transition: width .25s linear; }
    .settings .meter i.arriving { opacity: .75; background-size: 17px 17px; animation: settings-stripes .5s linear infinite; }
    .settings .meter i.others { background: var(--faint); opacity: .55; }
    .settings .meter.resetting i { display: none; }
    .settings .meter.resetting::before { content: ""; flex: 1; background: repeating-linear-gradient(-45deg, var(--warn) 0 6px, color-mix(in srgb, var(--warn) 45%, transparent) 6px 12px);
                                         background-size: 17px 17px; animation: settings-stripes .5s linear infinite; }
    @keyframes settings-stripes { to { background-position: 17px 0; } }
    @media (prefers-reduced-motion: reduce) { .settings .meter i, .settings .meter::before { animation: none !important; transition: none; } }
    .settings .legend { display: flex; flex-wrap: wrap; gap: 4px 14px; margin-top: 8px; font-size: 13px; color: var(--muted); }
    .settings .legend i, .settings .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }
    .settings .others { margin-top: 14px; display: grid; gap: 6px; }
    .settings .other { display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: baseline; font-size: 13.5px; padding: 8px 12px;
                       background: color-mix(in srgb, var(--chip) 55%, var(--panel)); border: 1px solid var(--line); border-radius: 10px; }
    .settings .other span { color: var(--muted); }
    .settings .list { display: grid; gap: 10px; margin-top: 10px; }
    .settings .model { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px 14px; align-items: start; padding: 13px 14px;
                       background: color-mix(in srgb, var(--chip) 55%, var(--panel)); border: 1px solid var(--line); border-radius: 12px; }
    .settings .model.on { border-color: color-mix(in srgb, var(--good) 55%, var(--line)); }
    .settings .model.wanted { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
    .settings .model h3 { margin: 0; font-size: 15.5px; letter-spacing: -.005em; }
    .settings .model p { margin: 3px 0 0; color: var(--muted); font-size: 13px; }
    .settings .model .actions { display: flex; flex-direction: column; gap: 6px; min-width: 96px; }
    .settings .pill { display: inline-block; border-radius: 999px; padding: 0 8px; font-size: 12px; font-weight: 500; border: 1px solid currentColor;
                      margin-left: 8px; vertical-align: 1px; color: var(--muted); }
    .settings .pill.loaded { color: var(--good); } .settings .pill.loading, .settings .pill.downloading { color: var(--warn); }
    .settings .pill.board { color: var(--accent); }
    .settings .graphs { display: flex; flex-wrap: wrap; gap: 5px 14px; margin-top: 8px; font-size: 13.5px; }
    .settings .graphs label { display: inline-flex; align-items: center; gap: 6px; }
    .settings .graphs span { color: var(--muted); font-variant-numeric: tabular-nums; }
    .settings .graphs span.there { color: var(--good); }
    .settings .graphs input { accent-color: var(--accent); margin: 0; }
    .settings details { margin-top: 8px; font-size: 13px; }
    .settings summary { cursor: pointer; color: var(--muted); }
    .settings .facts { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 2px 12px; margin: 8px 0 0; }
    .settings .facts dt { color: var(--muted); } .settings .facts dd { margin: 0; overflow-wrap: anywhere; }
    .settings .progress { margin-top: 9px; font-size: 12.5px; color: var(--muted); }
    .settings .progress .pbar { height: 7px; background: var(--track); border-radius: 999px; overflow: hidden; margin-bottom: 4px; }
    .settings .progress .pbar i { display: block; height: 100%; background: var(--spectrum); border-radius: 999px; }
    .settings .empty { border: 1px dashed var(--line); border-radius: 12px; padding: 14px; color: var(--muted); font-size: 13.5px; }
    .settings .confirm { margin-top: 12px; padding: 14px; border: 1px solid color-mix(in srgb, var(--warn) 55%, var(--line)); border-radius: 12px;
                         background: color-mix(in srgb, var(--warn) 7%, var(--panel)); }
    .settings .confirm p { margin: 0 0 8px; font-size: 13.5px; color: var(--muted); }
    .settings .confirm b { color: var(--ink); }
    .settings .confirm input { display: block; width: 100%; margin-top: 6px; padding: 7px 10px; border: 1px solid var(--line); border-radius: 8px;
                               background: var(--field); color: var(--ink); font: 13.5px var(--mono); }
    .settings .confirm code, .settings .about code { font: 12.5px var(--mono); color: var(--ink); background: var(--chip); border-radius: 6px; padding: 1px 6px; }
    .settings .about dl { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 8px 18px; margin: 0; }
    .settings .about dt { color: var(--muted); } .settings .about dd { margin: 0; }
    .settings .about img { width: 54px; height: 54px; border-radius: 14px; }
  ` }));

  let info = null, hub = null;      // the board's models, and the list on Hugging Face
  let mla = null;                   // the MLA's memory in full: who else is on it (GET /api/mla)
  let busy = null;                  // the model this window is acting on
  let said = "";                    // what the last action answered, when it was refused
  let wanted = null;                // a model to point at when the window opens
  let confirming = false, needsToken = false, resetSaid = "";
  let events = null, timer = null, drawn = "", drawnAt = 0;
  const chosen = {};                // model -> Set of graph sizes ticked for the next load
  const colour = (name) => COLOURS[Math.max(0, (info?.loaded || []).indexOf(name)) % COLOURS.length];

  // ---------------------------------------------------------------------------- the window
  const panels = {
    models: el("section", { className: "settings-panel", role: "tabpanel" }),
    add: el("section", { className: "settings-panel", role: "tabpanel" }),
    about: el("section", { className: "settings-panel about", role: "tabpanel" }),
  };
  const tabs = el("nav", { className: "settings-tabs", role: "tablist", ariaLabel: "Settings sections" });
  for (const [key, name] of [["models", "Models"], ["add", "Add Model"], ["about", "About"]]) {
    tabs.append(el("button", { type: "button", role: "tab", textContent: name, onclick: () => show(key) }));
    tabs.lastChild.dataset.tab = key;
  }
  const close = el("button", { className: "sbtn icon", type: "button", title: "Close", ariaLabel: "Close settings", textContent: "✕", onclick: shut });
  const root = el("div", { className: "settings", role: "dialog", ariaModal: "true", ariaLabel: "Settings" },
    el("div", { className: "settings-card" },
      el("div", { className: "settings-head" }, el("b", { textContent: "Settings" }), close),
      el("div", { className: "settings-body" }, tabs, el("div", { style: "min-height:0; display:grid" }, ...Object.values(panels)))));
  root.addEventListener("mousedown", (event) => { if (event.target === root) shut(); });
  addEventListener("keydown", (event) => { if (event.key === "Escape" && root.classList.contains("open")) { event.stopPropagation(); shut(); } }, true);

  function show(tab) {
    for (const button of tabs.children) button.classList.toggle("here", button.dataset.tab === tab);
    for (const [key, panel] of Object.entries(panels)) panel.classList.toggle("here", key === tab);
  }

  function open(tab = "models", model = null) {
    if (!root.isConnected) document.body.append(root);
    wanted = model;
    show(panels[tab] ? tab : "models");
    root.classList.add("open");
    refresh();
    listen();
    close.focus();
  }

  function shut() {
    root.classList.remove("open");
    confirming = false;
    events?.close(); events = null;
    clearTimeout(timer);
    if (location.pathname === "/models") history.replaceState(null, "", "/");
    else if (location.hash.startsWith("#settings")) history.replaceState(null, "", location.pathname + location.search);
    dispatchEvent(new Event("settings-closed"));       // a page may have a model to pick up
  }

  // ------------------------------------------------------------------------------- talking
  async function refresh() {
    if (!root.classList.contains("open")) return;
    let active = busy !== null;
    try {
      [info, hub, mla] = await Promise.all([fetch("/api/info").then((r) => r.json()), fetch("/api/hub").then((r) => r.json()),
                                           fetch("/api/mla").then((r) => r.json()).catch(() => null)]);
      render();
      active ||= Object.values(info.models).some((model) => model.progress) || !hub.listed
        || Object.values(hub.models).some((model) => model.state === "downloading");
    } catch (_) { /* board unreachable: keep the last view */ }
    clearTimeout(timer);
    timer = setTimeout(refresh, active ? 300 : 2500);
  }

  // The board also says what changes as it changes, so the memory bar follows a model in.
  function listen() {
    if (!window.EventSource || events) return;
    events = new EventSource("/api/events");
    events.onmessage = (event) => {
      info = JSON.parse(event.data);
      const shape = JSON.stringify(Object.entries(info.models).map(([name, model]) =>
        [name, model.state, model.loaded_seq_lens, model.resident_seq_lens, model.progress && [model.progress.stage, model.progress.step]]));
      if (shape !== drawn || (performance.now() - drawnAt > 300 && Object.values(info.models).some((model) => model.progress))) {
        drawn = shape; drawnAt = performance.now();
        render();
      } else {
        memory();
      }
    };
  }

  async function act(name, path, extra = {}) {
    busy = name; said = "";
    render();
    refresh();
    let body;
    try {
      body = await (await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify({ model: name, ...extra }) })).json();
    } catch (error) {
      body = { error: "The board did not answer: " + error.message };
    }
    busy = null;
    if (body.error) said = body.error;
    await refresh();
    render();
  }

  // --------------------------------------------------------------------------------- cards
  const button = (text, onclick, props = {}) => el("button", { type: "button", textContent: text, onclick, ...props, className: "sbtn " + (props.className || "") });

  function card(name, local, remote) {
    const known = CHECKPOINTS[local?.checkpoint] || [];
    const int8 = local && !remote && local.kind === "laya" && local.precision === "A_BF16_W_INT8";
    const title = remote?.title || (known[0] || local.checkpoint || name) + (int8 ? " INT8" : "");
    const about = remote?.about || known[1] || "";
    const precision = local?.precision || remote?.precision;
    const downloading = remote?.state === "downloading";
    const working = busy === name || local?.state === "loading" || local?.state === "unloading";
    const idle = busy === null && !working && !info.resetting;

    const pills = [];
    if (local?.state === "loading" || local?.state === "unloading") pills.push(el("span", { className: "pill loading", textContent: local.state + "…" }));
    else if (local?.loaded) pills.push(el("span", { className: "pill loaded", textContent: "loaded on the MLA" }));
    else if (local) pills.push(el("span", { className: "pill board", textContent: "on this board" }));
    if (downloading) pills.push(el("span", { className: "pill downloading", textContent: "downloading…" }));
    else if (!local) pills.push(el("span", { className: "pill", textContent: "on Hugging Face" }));

    // The graphs: tick boxes for what to load once it is on the board, a plain list before.
    const speed = Object.fromEntries((remote?.graphs || []).map((g) => [g.seq_len, g.latency_ms]));
    const graphs = el("div", { className: "graphs" });
    for (const graph of local ? local.graphs : remote.graphs) {
      const text = [mb(graph.bytes), graph.files > 1 ? `in ${graph.files} parts` : "", speed[graph.seq_len] ? `${speed[graph.seq_len]} ms` : ""].filter(Boolean).join(" · ");
      if (local) {
        // To begin with one length is ticked, since each is another copy of the weights: a Laya's
        // shortest, and CLM's longest, which is the one that takes a whole question in one pass.
        chosen[name] ??= new Set(local.loaded ? local.loaded_seq_lens : local.kind === "clm" ? local.seq_lens.slice(-1) : local.kind === "d1" && local.graphs.every((graph) => graph.files === 1) ? local.seq_lens : local.kind === "d1" ? local.seq_lens.slice(-1) : local.seq_lens.slice(0, 1));
        const box = el("input", { type: "checkbox", checked: local.loaded ? local.loaded_seq_lens.includes(graph.seq_len) : chosen[name].has(graph.seq_len),
                                  disabled: local.loaded || working });
        box.onchange = () => { box.checked ? chosen[name].add(graph.seq_len) : chosen[name].delete(graph.seq_len); render(); };
        const there = !local.loaded && (local.resident_seq_lens || []).includes(graph.seq_len);
        graphs.append(el("label", {}, box, `${graph.seq_len} tokens`, el("span", { textContent: text }),
                         there ? el("span", { className: "there", textContent: "on the MLA" }) : ""));
      } else {
        graphs.append(el("label", {}, `${graph.seq_len} tokens`, el("span", { textContent: text })));
      }
    }

    const facts = el("dl", { className: "facts" });
    const fact = (label, ...value) => { if (value[0]) facts.append(el("dt", { textContent: label }), el("dd", {}, ...value)); };
    fact("Encoder", remote?.card?.encoder || known[2]);
    fact("Languages", remote?.card?.languages || known[3]);
    fact("Precision", PRECISIONS[precision] || precision);
    fact("Accuracy", remote?.agreement ? `same decision as the PyTorch model in ${remote.agreement}` : "");
    const disk = local ? local.graphs.reduce((sum, g) => sum + g.bytes, 0) + local.fixed_bytes : 0;
    fact("Size", local ? `${mb(disk)} on this board's disk` : `${mb(remote.bytes)} to download`
                       + (remote.have_bytes && !downloading ? `, ${mb(remote.have_bytes)} of it already here` : ""));
    if (remote && hub?.repo) {
      const upstream = /^[\w.-]+\/[\w.-]+$/.test(remote.card?.upstream || "") ? remote.card.upstream : null;
      fact("Source", el("a", { href: `${hub.page}/tree/main/${remote.path}`, target: "_blank", rel: "noopener", textContent: `${hub.repo}/${remote.path}` }),
        ...(upstream ? [", compiled from ", el("a", { href: `https://huggingface.co/${upstream}`, target: "_blank", rel: "noopener", textContent: upstream })] : []),
        remote.card?.license ? ` (${remote.card.license})` : "");
    } else {
      fact("Source", "compiled and copied to this board; not on Hugging Face",
           ...(local.kind === "clm" ? [", from ", el("a", { href: "https://huggingface.co/Contrastive-LM/CLM-v0.1-8B", target: "_blank", rel: "noopener", textContent: "Contrastive-LM/CLM-v0.1-8B" }), " (Apache-2.0)"] : []),
           ...(local.kind === "d1" ? [", from ", el("a", { href: `https://huggingface.co/LiquidAI/${local.checkpoint}`, target: "_blank", rel: "noopener", textContent: `LiquidAI/${local.checkpoint}` }),
                                      " (LFM Open License v1.0: commercial use is limited to organisations under 10 million dollars of annual revenue)"] : []));
    }

    // What it takes or would take on the MLA, said where the Load button is decided.
    let mla = "";
    if (local) {
      const size = (lens) => local.graphs.filter((g) => lens.includes(g.seq_len)).reduce((sum, g) => sum + g.bytes, 0);
      mla = local.loaded ? `${mb(size(local.loaded_seq_lens))} on the MLA` : `the ticked graphs take ${mb(size([...chosen[name]]))} on the MLA`;
    }

    // Progress comes from the server: stage boundaries for a load, bytes for a download.
    let progress = "";
    if (local?.progress) {
      const percent = Math.round(local.progress.fraction * 100);
      const fill = el("i"); fill.style.width = percent + "%";
      progress = el("div", { className: "progress", role: "progressbar", ariaValueMin: 0, ariaValueMax: 100, ariaValueNow: percent, ariaLabel: `${local.state} ${title}` },
        el("div", { className: "pbar" }, fill),
        el("span", { textContent: `${local.progress.stage} · step ${local.progress.step} of ${local.progress.steps} · ${percent}%` }));
    } else if (downloading && remote.progress) {
      const p = remote.progress, percent = Math.round(p.fraction * 100);
      const fill = el("i"); fill.style.width = percent + "%";
      progress = el("div", { className: "progress", role: "progressbar", ariaValueMin: 0, ariaValueMax: 100, ariaValueNow: percent, ariaLabel: `downloading ${title}` },
        el("div", { className: "pbar" }, fill),
        el("span", { textContent: `${mb(p.done_bytes)} of ${mb(p.total_bytes)} · ${percent}%`
          + (p.bytes_per_second ? ` · ${(p.bytes_per_second / (1 << 20)).toFixed(1)} MB/s · ${left(p.seconds_left)}` : "") + (p.file ? ` · ${p.file}` : "") }));
    }
    const problem = remote?.error ? el("p", { className: "said", textContent: remote.error }) : "";

    const actions = el("div", { className: "actions" });
    if (local?.loaded) {
      actions.append(button("Unload", () => act(name, "/api/models/unload"), { disabled: !idle }));
    } else if (local) {
      actions.append(button("Load", () => act(name, "/api/models/load", { seq_lens: [...chosen[name]].sort((a, b) => a - b) }),
                            { className: "primary", disabled: !idle || !chosen[name].size }));
      actions.append(button("Delete", () => remove(name, title, disk, Boolean(remote)),
                            { className: "danger", disabled: !idle, title: "Remove this model's files from the board's disk" }));
    } else if (downloading) {
      actions.append(button("Cancel", () => act(name, "/api/hub/cancel"), { disabled: busy !== null }));
    } else {
      const another = Object.values(hub.models).some((model) => model.state === "downloading");
      actions.append(button(remote.have_bytes ? "Continue" : "Download", () => act(name, "/api/hub/download"),
                            { className: "primary", disabled: !idle || another }));
      if (remote.have_bytes) {
        actions.append(button("Discard", () => act(name, "/api/models/delete"),
                              { className: "danger", disabled: !idle, title: "Remove the part that was downloaded" }));
      }
    }
    const dot = local?.loaded ? el("i", { className: "dot" }) : "";
    if (dot) dot.style.background = colour(name);
    return el("div", { className: "model" + (local?.loaded ? " on" : "") + (wanted === name ? " wanted" : ""), id: "settings-" + name },
      el("div", {}, el("h3", {}, dot, title, ...pills), el("p", { textContent: [about, mla].filter(Boolean).join(" · ") }),
         graphs, progress, problem, el("details", {}, el("summary", { textContent: "Model Card" }), facts)),
      actions);
  }

  // Deleting removes files from the board, so it is asked about first.
  function remove(name, title, bytes, downloadable) {
    const again = downloadable ? "It can be downloaded again from Hugging Face."
                               : "It is not on Hugging Face: getting it back means compiling or copying it again.";
    if (confirm(`Delete ${title} from this board's disk (${mb(bytes)})?\n\n${again}`)) act(name, "/api/models/delete");
  }

  // -------------------------------------------------------------------------------- memory
  const meter = el("div", { className: "meter", role: "img" }), memoryText = el("span", { className: "note", style: "margin:0" });
  const legend = el("div", { className: "legend" });

  // The memory bar: a segment a loaded model in its colour, what is still arriving striped,
  // and in grey what the accelerator holds besides. Drawn as often as the board reports.
  function memory() {
    if (!info) return;
    const { mla_loaded_bytes: used, mla_arriving_bytes: arriving = 0, mla_total_bytes: total, mla_held_bytes: held = null } = info.memory;
    meter.classList.toggle("resetting", Boolean(info.resetting));
    if (info.resetting) { memoryText.textContent = "Resetting the accelerator: every model is coming off the MLA…"; return; }
    // With a reading of what is held, the figures are measured: this app's share is what the
    // accelerator took on as its models were loaded. Without one they are the graph files' sizes.
    const known = held !== null && total && mla && mla.held_bytes !== null;
    const mine = known ? mla.app_bytes : used + arriving, beyond = known ? Math.max(held - mine, 0) : 0;
    memoryText.textContent = known
      ? `${mb(held)} held of ${mb(total)} · this app ${mb(mine)} · other programs ${mb(beyond)} · ${mb(Math.max(total - held, 0))} free`
      : `${mb(used + arriving)} from this app` + (total ? ` of ${mb(total)}` : "");
    // A model's segment is its share of this app's part, by the size of its graphs.
    const scale = known && used + arriving > 0 ? Math.min(1, mine / (used + arriving)) : 1;
    const share = (bytes) => (total ? bytes * scale / total * 100 : 0).toFixed(2) + "%";
    const parts = [], keys = [];
    for (const [name, model] of Object.entries(info.models)) {
      if (!model.mla_bytes) continue;
      const part = el("i", { className: model.loaded ? "" : "arriving", title: `${name}: ${mb(model.mla_bytes)}` });
      part.style.width = share(model.mla_bytes);
      part.style.background = model.loaded ? colour(name)
        : `repeating-linear-gradient(-45deg, var(--accent) 0 6px, color-mix(in srgb, var(--accent) 45%, transparent) 6px 12px)`;
      parts.push(part);
      const key = el("span", {}, el("i"), `${CHECKPOINTS[model.checkpoint]?.[0] || name} ${mb(model.mla_bytes)}`);
      key.firstChild.style.background = model.loaded ? colour(name) : "var(--accent)";
      keys.push(key);
    }
    if (known && beyond > (64 << 20)) {
      const others = el("i", { className: "others", title: "Held for other programs, or not given back yet" });
      others.style.width = (beyond / total * 100).toFixed(2) + "%";
      parts.push(others);
      const key = el("span", {}, el("i"), `other programs, or not given back ${mb(beyond)}`);
      key.firstChild.style.background = "var(--faint)";
      keys.push(key);
    }
    meter.replaceChildren(...parts);
    legend.replaceChildren(...keys);
  }

  // Who else is on the accelerator. What is held beyond this app's models is measured; whose
  // it is, the accelerator does not say, so each program is listed with what it reports itself.
  function others() {
    if (!mla || mla.held_bytes === null) return "";
    const rows = [];
    for (const program of mla.programs) {
      const said = program.reported;
      const models = said ? Object.entries(said.models).map(([name, bytes]) => `${name} ${mb(bytes)}`).join(", ") : "";
      rows.push(el("div", { className: "other" }, el("b", { textContent: program.name }),
        el("span", { textContent: said ? (said.bytes ? `reports ${mb(said.bytes)}: ${models}` : "reports no model loaded") : "connected; it does not say what it has loaded" })));
    }
    if (!rows.length) rows.push(el("div", { className: "other" }, el("span", { textContent: "No other program is connected to the accelerator." })));
    const beyond = mla.other_bytes;
    return el("div", { className: "others" },
      el("div", { className: "row" }, el("span", { className: "label grow", textContent: "Other Programs" }),
         el("span", { className: "note", style: "margin:0", textContent: `${mb(beyond)} held beyond this app's models` + (mla.app_measured ? "" : " (estimated)") })),
      ...rows,
      el("p", { className: "note", textContent: "The accelerator says how much is held in all, not by whom. A program's own figure is its estimate, "
        + "and it can name models that a reset has since taken off; memory held with no program left to claim it has not been given back." }));
  }

  // Resetting the accelerator. The board itself may do it freely; a browser anywhere else is
  // asked for the reset token, which is remembered for as long as this tab is open.
  const storedToken = () => { try { return sessionStorage.getItem("laya-reset-token") || ""; } catch (_) { return ""; } };
  async function reset(token) {
    resetSaid = "";
    let body;
    try {
      body = await (await fetch("/api/mla/reset", { method: "POST", headers: { "Content-Type": "application/json" },
                                                    body: JSON.stringify(token ? { token } : {}) })).json();
    } catch (error) {
      body = { error: "The board did not answer: " + error.message };
    }
    if (body.needs_token) { needsToken = true; resetSaid = token ? "That is not this board's reset token." : ""; }
    else if (body.error) resetSaid = body.error;
    else {
      if (token) { try { sessionStorage.setItem("laya-reset-token", token); } catch (_) { /* asked again next time */ } }
      confirming = false;
      info = body;
    }
    render();
  }

  function resetBlock() {
    const go = button("Reset Accelerator", () => { confirming = true; resetSaid = ""; render(); },
                      { disabled: Boolean(info.resetting) || confirming, title: "Restart the MLA services and take every model off the accelerator" });
    if (info.resetting) go.textContent = "Resetting…";
    const block = el("div", { className: "block" },
      el("div", { className: "row" }, go, el("span", { className: "note", style: "margin:0",
        textContent: "Frees memory the accelerator has not given back. The app never does this on its own." })));
    if (confirming) {
      const token = el("input", { autocomplete: "off", spellcheck: false, ariaLabel: "Reset token", value: storedToken() });
      const yes = button("Reset Accelerator", () => { yes.disabled = true; yes.textContent = "Resetting…"; reset(token.value.trim()); }, { className: "danger" });
      block.append(el("div", { className: "confirm" },
        el("p", {}, el("b", { textContent: "Reset the Accelerator? " }), "This restarts the board's MLA services. Every model comes off the accelerator: ",
           "this app's, and any other application's, such as Neat GenAI Studio, which then has to load its models again. Nothing on disk is touched."),
        needsToken ? el("p", {}, "This browser is not on the board, so the reset token is needed. On the board, ", el("code", { textContent: "./run.sh --reset-token" }), " prints it.", token) : "",
        resetSaid ? el("p", { className: "said", style: "margin:0 0 8px", textContent: resetSaid }) : "",
        el("div", { className: "row" }, yes, button("Cancel", () => { confirming = false; render(); }))));
    }
    return block;
  }

  // ------------------------------------------------------------------------------- drawing
  function render() {
    if (!info) return;
    const remote = hub?.models || {};
    const local = Object.keys(info.models), loaded = info.loaded || local.filter((name) => info.models[name].loaded);
    const more = Object.keys(remote).filter((name) => !(name in info.models));
    memory();

    // Keep what the user has open (a model card, the scroll position) across a redraw.
    const opened = new Set([...root.querySelectorAll("details[open]")].map((d) => d.closest(".model")?.id));
    const scroll = Object.fromEntries(Object.entries(panels).map(([key, panel]) => [key, panel.scrollTop]));

    const unloadAll = button("Unload All", () => act("*", "/api/models/unload", { model: loaded[0], all: true }),
                             { disabled: busy !== null || loaded.length < 2, title: "Take every model of this app off the accelerator" });
    panels.models.replaceChildren(
      el("div", { className: "block" },
        el("div", { className: "label", textContent: "Load Status" }),
        el("p", { className: "note", textContent: `${loaded.length} loaded · ${local.length} on this board`
          + (more.length ? ` · ${more.length} more on Hugging Face` : "") }),
        said ? el("p", { className: "said", textContent: said }) : "",
        el("p", { className: "note", textContent: "Several models can be loaded at once, as many as fit in the MLA's memory. "
          + "The pages then let you choose which answers, and Compare asks them all." })),
      el("div", { className: "block" },
        el("div", { className: "row" }, el("span", { className: "label grow", textContent: "MLA Memory" }), memoryText),
        meter, legend,
        el("p", { className: "note", textContent: "Compiled graphs live in memory reserved for the MLA, separate from the board's Linux RAM "
          + "and shared with every program that uses the accelerator. Grey is what it holds besides this app's models." }),
        others()),
      resetBlock(),
      el("div", { className: "row" }, el("span", { className: "label grow", textContent: "Downloaded — on This Board" }), loaded.length > 1 ? unloadAll : ""),
      el("div", { className: "list" }, ...(local.length ? [...loaded, ...local.filter((name) => !loaded.includes(name))].map((name) => card(name, info.models[name], remote[name]))
        : [el("div", { className: "empty" }, "No model is on this board yet. ", button("Add Model", () => show("add")))])));

    const disk = hub?.repo ? el("p", { className: "note" }, "Models come from ",
      el("a", { href: hub.page, target: "_blank", rel: "noopener", textContent: hub.repo }),
      " on Hugging Face onto this board's disk", ...(hub.directory ? [", in ", el("code", { textContent: hub.directory })] : []),
      `, which has ${mb(hub.free_bytes)} free.`) : "";
    panels.add.replaceChildren(
      el("div", { className: "block" }, el("div", { className: "label", textContent: "Available to Download — from Hugging Face" }), disk,
        hub?.error ? el("p", { className: "said", textContent: hub.error }) : "",
        said ? el("p", { className: "said", textContent: said }) : ""),
      el("div", { className: "list", style: "margin-top:0" }, ...(more.length ? more.map((name) => card(name, undefined, remote[name]))
        : [el("div", { className: "empty", textContent: !hub ? "Reading the list…" : !hub.repo ? "This app was started without a Hugging Face repository (--hub none)."
            : !hub.listed && !hub.error ? "Reading the list from Hugging Face…" : "Every model published there is already on this board." })])));

    panels.about.replaceChildren(
      el("div", { className: "row", style: "margin-bottom:16px" }, el("img", { src: "/static/brand/neat-decision.svg", alt: "" }),
        el("div", {}, el("b", { style: "font-size:18px", textContent: "Neat Decision Studio" }),
           el("p", { className: "note", style: "margin:0", textContent: "System-1 decision models on the SiMa.ai Modalix MLSoC" }))),
      el("dl", {},
        el("dt", { textContent: "Version" }), el("dd", { textContent: info.version || "unknown" }),
        el("dt", { textContent: "Update" }), el("dd", {}, "On the board: ", el("code", { textContent: "neat-decision update" })),
        el("dt", { textContent: "Models" }), el("dd", { textContent: `${loaded.length} loaded, ${local.length} on this board` }),
        el("dt", { textContent: "Source" }), el("dd", {}, el("a", { href: "https://github.com/dotimothy/neat-decision-studio", target: "_blank", rel: "noopener",
                                                                   style: "color:var(--accent)", textContent: "github.com/dotimothy/neat-decision-studio" })),
        el("dt", { textContent: "Showcase" }), el("dd", {}, el("a", { href: "/showcase", style: "color:var(--accent)", textContent: "The Story of the Demo, as Slides" }))));

    for (const id of opened) { const details = id && document.getElementById(id)?.querySelector("details"); if (details) details.open = true; }
    for (const [key, panel] of Object.entries(panels)) panel.scrollTop = scroll[key];
    if (wanted) { document.getElementById("settings-" + wanted)?.scrollIntoView({ block: "center" }); wanted = null; }
  }

  window.openSettings = open;
  // /models is the app with Settings open; so is #settings, or #settings=add, on any page.
  const opener = () => {
    if (location.pathname === "/models") open("models", location.hash.slice(1) || null);
    else if (location.hash.startsWith("#settings")) open(location.hash.split("=")[1] || "models");
  };
  document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", opener) : opener();
  addEventListener("hashchange", opener);
})();
