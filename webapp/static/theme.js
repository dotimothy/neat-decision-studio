// Dark mode, on or off, whatever the system says.
//
// The pages follow the system through `@media (prefers-color-scheme: dark)`: in neat.css, in
// each page's own styles and in the styles some scripts add. So that none of them has to be
// written twice, a choice made with the header's button is put into force by rewriting those
// rules' conditions where they stand: "dark" makes every dark rule apply and every light one
// not, "light" the other way, and with no choice made the rules are as they were written.
// The choice is kept in the browser (localStorage), so every page opens with it.
//
//   theme.current()          "dark" or "light": what is showing
//   theme.toggle()           the other one, kept from now on
//   theme.set("dark" | "light" | null)     null: back to the system's
//
// A "themechange" event on the window says when what is showing has changed, for pages that
// draw with colours read from the styles (the games' canvases).
"use strict";
(() => {
  const KEY = "laya.theme";
  const system = matchMedia("(prefers-color-scheme: dark)");
  const chosen = () => { try { const kept = localStorage.getItem(KEY); return kept === "dark" || kept === "light" ? kept : null; } catch (_) { return null; } };
  const current = () => chosen() || (system.matches ? "dark" : "light");
  const written = new WeakMap();      // a media rule -> its condition as it was written
  const FEATURE = /\(\s*prefers-color-scheme\s*:\s*(dark|light)\s*\)/g;

  function rewrite(rules, forced) {
    for (const rule of rules) {
      if (rule instanceof CSSMediaRule) {
        const own = written.get(rule) ?? rule.media.mediaText;
        if (FEATURE.test(own)) {
          written.set(rule, own);
          // A condition that always holds, or never does, in place of the question about the
          // system; anything else in the rule's condition (a width, say) stays.
          const wanted = forced === null ? own
            : own.replace(FEATURE, (_, scheme) => scheme === forced ? "(min-width: 0px)" : "(max-width: 0px) and (min-width: 1px)");
          if (rule.media.mediaText !== wanted) rule.media.mediaText = wanted;
        }
        FEATURE.lastIndex = 0;
      }
      if (rule.cssRules) rewrite(rule.cssRules, forced);
    }
  }

  function apply() {
    const forced = chosen();
    for (const sheet of document.styleSheets) {
      try { rewrite(sheet.cssRules, forced); } catch (_) { /* a sheet from elsewhere: not ours to read */ }
    }
    const root = document.documentElement;
    root.dataset.theme = current();
    root.style.colorScheme = current();        // form controls and scrollbars follow
  }

  function set(theme) {
    const before = current();
    try { theme ? localStorage.setItem(KEY, theme) : localStorage.removeItem(KEY); } catch (_) { /* this page only */ }
    apply();
    if (current() !== before) dispatchEvent(new Event("themechange"));
  }

  window.theme = { current, chosen, set, toggle: () => set(current() === "dark" ? "light" : "dark") };
  apply();
  // Styles that arrive later (a script's own, a sheet still loading) are brought into line too.
  new MutationObserver(apply).observe(document.documentElement, { childList: true, subtree: true, attributes: false, characterData: false });
  addEventListener("load", apply);
  system.addEventListener("change", () => { if (!chosen()) { apply(); dispatchEvent(new Event("themechange")); } });
  // Another tab changed it: follow.
  addEventListener("storage", (event) => { if (event.key === KEY) { apply(); dispatchEvent(new Event("themechange")); } });
})();
