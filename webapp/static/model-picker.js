// Shared by every page except the model manager: which model is answering, and the token
// budget for its requests.
//
// Choosing and loading a model is the model manager's job (the Models page), and one model is
// on the MLA at a time. So a page only names the loaded model and uses it; with none loaded it
// says so, pointing at the Models tab, and it starts by itself once a model is there.
//
//   const picker = createModelPicker({ mount, onReady: (ready) => ... });
//   picker.name        the loaded model, to send as "model" in a request (null if none)
//   picker.budget      the token budget, to send as "max_len"
//   picker.ready       whether a model is loaded
//   picker.lost()      call when a request came back "not loaded"
//   picker.report(u)   call with a response's `usage`, to show what the budget did
//   picker.onBudget    set to a function to hear about a changed budget
//
// `onReady(false)` then `onReady(true)` is also what a page sees when the manager swaps the
// model for another, so a page starts over with the new one.
//
// The token budget is how many tokens a question, its options and the state may take together.
// Its ceiling is the largest graph the model has loaded.
"use strict";

const CHECKPOINT_TITLES = {
  "laya": "Laya (English, general)",
  "laya-typed-decisions": "Laya typed-decisions",
  "laya-multilingual": "Laya multilingual",
  "laya-dino": "Laya-dino (dino game head)",
};

function createModelPicker({ mount, onReady, madeFor }) {
  if (!document.getElementById("picker-style")) {
    document.head.append(Object.assign(document.createElement("style"), { id: "picker-style", textContent: `
      .picker { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-left: auto; font-size: 14px; }
      .picker label { color: var(--muted); }
      .picker .picker-model { font-weight: 650; }
      .picker .picker-note { color: var(--muted); font-size: 13px; }
      .picker .picker-model.none { color: var(--warn); }
      .picker input[type=range] { width: 110px; margin: 0; accent-color: var(--accent); }
      .picker input[type=number] { width: 68px; font: inherit; color: var(--ink); background: var(--field);
                                   border: 1px solid var(--line); border-radius: 8px; padding: 4px 6px; }
    ` }));
  }
  const make = (tag, props = {}) => Object.assign(document.createElement(tag), props);
  const model = make("span", { className: "picker-model" });
  const note = make("span", { className: "picker-note" });
  const slider = make("input", { type: "range", min: 16, max: 128, step: 1, value: 128, ariaLabel: "Token budget" });
  const number = make("input", { type: "number", min: 16, max: 128, step: 1, value: 128, ariaLabel: "Token budget, exact" });
  const used = make("span", { className: "picker-note" });
  mount.classList.add("picker");
  mount.replaceChildren(make("label", { textContent: "Model" }), model, note,
                        make("label", { textContent: "Token budget" }), slider, number, used);

  const picker = { name: null, ready: false, models: {}, budget: 0, onBudget: null };
  const title = (name) => (CHECKPOINT_TITLES[picker.models[name]?.checkpoint] || name)
    + (picker.models[name]?.precision === "A_BF16_W_INT8" ? ", INT8 weights" : "");
  let touched = false, active = false;

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

  function apply(info) {
    picker.models = info.models;
    const loaded = Object.keys(info.models).find((name) => info.models[name].loaded) || null;
    const loading = Object.keys(info.models).find((name) => info.models[name].state === "loading");
    if (loaded !== picker.name && picker.ready) { picker.ready = false; picker.name = null; onReady(false); }   // gone, or swapped
    picker.name = loaded;
    const percent = loading && info.models[loading].progress ? ` ${Math.round(info.models[loading].progress.fraction * 100)}%` : "";
    model.textContent = loaded ? title(loaded) : loading ? `${title(loading)}, loading…${percent}` : "none on the MLA";
    active = Boolean(loading);
    model.classList.toggle("none", !loaded);
    note.textContent = loaded ? (madeFor && loaded !== madeFor ? `This game was set up for ${title(madeFor)}.` : "")
      : loading ? "" : "Load one in the Models tab.";
    if (!loaded) used.textContent = "";
    showBudget();
    if (loaded && !picker.ready) { picker.ready = true; onReady(true); }
  }

  picker.refresh = async () => { try { apply(await (await fetch("/api/info")).json()); } catch (_) { /* keep the last view */ } };
  picker.lost = () => picker.refresh();

  // Notice when the manager loads, unloads or swaps a model; look often while one is loading.
  (async function poll() { await picker.refresh(); setTimeout(poll, active ? 250 : 2000); })();
  return picker;
}
