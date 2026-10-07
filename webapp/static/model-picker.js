// Shared by every page except the model manager: which model is answering, and the token
// budget for its requests.
//
// Loading a model is the model manager's job (Settings, behind the gear), and several can be
// on the MLA at once. So a page chooses among the loaded models: with one loaded it names it,
// with several it offers them, and with none it says so, pointing at Settings, and starts by
// itself once a model is there. A page's choice is kept for the browser session.
//
//   const picker = createModelPicker({ mount, onReady: (ready) => ... });
//   picker.name        the model answering, to send as "model" in a request (null if none)
//   picker.budget      the token budget, to send as "max_len"
//   picker.ready       whether a model is loaded
//   picker.lost()      call when a request came back "not loaded"
//   picker.report(u)   call with a response's `usage`, to show what the budget did
//   picker.onBudget    set to a function to hear about a changed budget
//
// A game has seats, and any loaded model can sit in one, so that two different models play
// each other. A seat is a <select> with an <option value="laya">; `picker.seats(select)` keeps
// that option as one per loaded model (each still with the value "laya", so a game's own
// handling of the seat is unchanged), and `picker.seatModel(select)` says which model the
// chosen one is. `picker.short(name)` is a model's name short enough for a seat, and
// `picker.budgetFor(name)` the token budget to send it (the page's own budget is for the
// page's model; another model takes its own).
//
// `onReady(false)` then `onReady(true)` is also what a page sees when its model is unloaded
// while another is there, or when another is chosen, so a page starts over with the new one.
//
// The token budget is how many tokens a question, its options and the state may take together.
// Its ceiling is the largest graph the model has loaded.
"use strict";

const CHECKPOINT_SHORT = {
  "CLM-v0.1-8B": "CLM", "laya": "Laya", "laya-typed-decisions": "Laya Typed-Decisions", "laya-multilingual": "Laya Multilingual",
  "laya-dino": "Laya-dino", "laya-chess": "Laya-chess",
};

const CHECKPOINT_TITLES = {
  "CLM-v0.1-8B": "CLM v0.1 8B (Qwen3-8B encoder)",
  "laya": "Laya (English, general)",
  "laya-typed-decisions": "Laya typed-decisions",
  "laya-multilingual": "Laya multilingual",
  "laya-dino": "Laya-dino (dino game head)",
  "laya-chess": "Laya-chess",
};

function createModelPicker({ mount, onReady, madeFor }) {
  if (!document.getElementById("picker-style")) {
    document.head.append(Object.assign(document.createElement("style"), { id: "picker-style", textContent: `
      .picker { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-left: auto; font-size: 14px; }
      .picker label { color: var(--muted); }
      .picker .picker-model { font-weight: 650; }
      .picker .picker-note { color: var(--muted); font-size: 13px; }
      .picker .picker-model.none { color: var(--warn); }
      .picker select.picker-choice { font: inherit; font-weight: 650; max-width: 250px; }
      .picker a.picker-settings { color: var(--accent); cursor: pointer; text-decoration: none; }
      .picker a.picker-settings:hover { text-decoration: underline; }
      .picker input[type=range] { width: 110px; margin: 0; accent-color: var(--accent); }
      .picker input[type=number] { width: 68px; font: inherit; color: var(--ink); background: var(--field);
                                   border: 1px solid var(--line); border-radius: 8px; padding: 4px 6px; }
    ` }));
  }
  const make = (tag, props = {}) => Object.assign(document.createElement(tag), props);
  const model = make("span", { className: "picker-model" });
  const choice = make("select", { className: "picker-choice", ariaLabel: "Model", hidden: true });
  const settings = make("a", { className: "picker-settings", textContent: "Open Settings", hidden: true });
  settings.onclick = () => window.openSettings?.("models", madeFor || null);
  const note = make("span", { className: "picker-note" });
  const slider = make("input", { type: "range", min: 16, max: 128, step: 1, value: 128, ariaLabel: "Token budget" });
  const number = make("input", { type: "number", min: 16, max: 128, step: 1, value: 128, ariaLabel: "Token budget, exact" });
  const used = make("span", { className: "picker-note" });
  mount.classList.add("picker");
  mount.replaceChildren(make("label", { textContent: "Model" }), model, choice, note, settings,
                        make("label", { textContent: "Token budget" }), slider, number, used);

  const picker = { name: null, ready: false, models: {}, loaded: [], budget: 0, onBudget: null };
  const title = (name) => (CHECKPOINT_TITLES[picker.models[name]?.checkpoint] || name)
    + (picker.models[name]?.precision === "A_BF16_W_INT8" && picker.models[name]?.kind !== "clm" ? ", INT8 weights" : "");
  let touched = false, active = false;

  // Seats: see the top of the file.
  const seated = [];
  picker.short = (name) => (CHECKPOINT_SHORT[picker.models[name]?.checkpoint] || name || "Laya")
    + (picker.models[name]?.precision === "A_BF16_W_INT8" && picker.models[name]?.kind !== "clm" ? " INT8" : "");
  picker.seatModel = (select) => {
    const model = select.selectedOptions[0]?.dataset.model;
    return picker.loaded.includes(model) ? model : picker.name;
  };
  picker.budgetFor = (name) => name === picker.name ? picker.budget : 0;
  function fillSeat(select) {
    const all = picker.loaded, mark = all.join("\n") + "|" + picker.name;
    if (select.dataset.filled === mark) return;
    select.dataset.filled = mark;
    const chosen = select.selectedOptions[0], kind = chosen ? chosen.value : select.value;
    const model = all.includes(chosen?.dataset.model) ? chosen.dataset.model : picker.name;
    const old = [...select.options].filter((option) => option.value === "laya");
    const words = select.dataset.seat || "{}";              // how this seat names a model, e.g. "Race {}"
    const made = (all.length ? all : [null]).map((name) => {
      const option = make("option", { value: "laya", textContent: words.replace("{}", name ? picker.short(name) : "Laya") });
      if (name) option.dataset.model = name;
      return option;
    });
    if (old.length) old[0].before(...made); else select.append(...made);
    for (const option of old) option.remove();
    if (kind === "laya") (made.find((option) => option.dataset.model === model) || made[0]).selected = true;
    else select.value = kind;
  }
  picker.seats = (...selects) => { for (const select of selects) { seated.push(select); fillSeat(select); } };

  // The budget follows its ceiling until the user sets their own, and never exceeds it.
  function showBudget() {
    const info = picker.models[picker.name];
    slider.disabled = number.disabled = !info;
    if (!info) return;
    const ceiling = Math.min(Math.max(...info.loaded_seq_lens), info.max_len || Infinity);
    slider.max = number.max = ceiling;
    if (!touched || Number(number.value) > ceiling) number.value = ceiling;
    slider.value = number.value;
    picker.budget = Number(number.value);
  }
  function setBudget(value) {
    touched = true;
    number.value = Math.max(Number(number.min), Math.min(Number(number.max), Math.round(Number(value) || 0)));
    showBudget();
    used.textContent = "";
    if (picker.onBudget) picker.onBudget();
  }
  slider.oninput = () => setBudget(slider.value);
  number.onchange = () => setBudget(number.value);
  picker.report = (usage) => {
    used.textContent = `${usage.tokens} used` + (usage.truncated ? `, ${usage.state_tokens_dropped} cut from the state` : "");
  };

  // Which of the loaded models this page uses: the one chosen here before, else the one the
  // page was made for, else the first general one that was loaded.
  const KEPT = "laya.model:" + location.pathname;
  const kept = () => { try { return sessionStorage.getItem(KEPT); } catch (_) { return null; } };
  const keep = (name) => { try { sessionStorage.setItem(KEPT, name); } catch (_) { /* not kept */ } };
  let options = "";
  choice.onchange = () => { keep(choice.value); picker.refresh(); };

  function apply(info) {
    picker.models = info.models;
    const all = info.loaded || Object.keys(info.models).filter((name) => info.models[name].loaded);
    picker.loaded = all;
    const loaded = all.includes(picker.name) && kept() === picker.name ? picker.name
      : [kept(), picker.name, madeFor].find((name) => all.includes(name))
        // A page made for no model in particular starts on one that is not a game's own.
        // Among those a Laya before CLM: the pages are worded for Laya, and it answers at once.
        || all.find((name) => !/chess|dino/.test(info.models[name].checkpoint || "") && info.models[name].kind !== "clm")
        || all.find((name) => !/chess|dino/.test(info.models[name].checkpoint || "")) || all[0] || null;
    const loading = Object.keys(info.models).find((name) => info.models[name].state === "loading");
    if (loaded !== picker.name && picker.ready) { picker.ready = false; picker.name = null; onReady(false); }   // gone, or another chosen
    picker.name = loaded;
    const percent = loading && info.models[loading].progress ? ` ${Math.round(info.models[loading].progress.fraction * 100)}%` : "";
    // One loaded: its name. Several: a list to choose from.
    choice.hidden = all.length < 2;
    model.hidden = !choice.hidden;
    if (!choice.hidden) {
      if (options !== all.join("\n")) {
        options = all.join("\n");
        choice.replaceChildren(...all.map((name) => make("option", { value: name, textContent: title(name) })));
      }
      choice.value = loaded;
    }
    model.textContent = loaded ? title(loaded) : loading ? `${title(loading)}, loading…${percent}` : "none on the MLA";
    active = Boolean(loading);
    model.classList.toggle("none", !loaded);
    note.textContent = loaded ? (madeFor && loaded !== madeFor ? `This game was set up for ${title(madeFor)}.` : "")
      : loading ? "" : "Load one in Settings.";
    settings.hidden = Boolean(loaded) && !(madeFor && loaded !== madeFor) || Boolean(loading && !loaded);
    if (!loaded) used.textContent = "";
    showBudget();
    for (const select of seated) fillSeat(select);
    if (loaded && !picker.ready) { picker.ready = true; onReady(true); }
  }

  picker.refresh = async () => { try { apply(await (await fetch("/api/info")).json()); } catch (_) { /* keep the last view */ } };
  picker.lost = () => picker.refresh();

  // Settings closing is when a model is most likely to have changed.
  addEventListener("settings-closed", () => picker.refresh());
  // Notice when the manager loads or unloads a model; look often while one is loading.
  (async function poll() { await picker.refresh(); setTimeout(poll, active ? 250 : 2000); })();
  return picker;
}
