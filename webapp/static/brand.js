// The brand header and footer every page shares, in the style of SiMa.ai's Neat demos.
//
// A page writes a plain <header> holding its title (h1), a tagline (p) and the tab links (nav).
// This rebuilds it as the Neat app bar (the mark, the product line and the tabs), moves the
// page's own title below it, and adds the footer. The landing page (`<body data-home>`) opens
// with its own headline, so it gets the bar and the footer only.
"use strict";
(() => {
  const make = (tag, props = {}, ...children) => {
    const node = Object.assign(document.createElement(tag), props);
    node.append(...children);
    return node;
  };
  // Full screen, for showing the demo on a display. A page change leaves full screen (each
  // page is its own document, and a browser only enters it on a click or a key), so the
  // choice is remembered for the session and taken up again at the first click or key on
  // the next page.
  const REMEMBER = "laya.fullscreen";
  const recall = () => { try { return sessionStorage.getItem(REMEMBER) === "1"; } catch (_) { return false; } };
  const remember = (on) => { try { on ? sessionStorage.setItem(REMEMBER, "1") : sessionStorage.removeItem(REMEMBER); } catch (_) { /* not kept */ } };
  function fullscreenButton() {
    if (!document.fullscreenEnabled) return "";
    const icons = {   // corners pointing out, and pointing in
      enter: "M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5",
      leave: "M9 4v5H4M20 9h-5V4M15 20v-5h5M4 15h5v5",
    };
    const button = make("button", { className: "header-button", type: "button" });
    const show = () => {
      const on = Boolean(document.fullscreenElement);
      button.title = on ? "Exit Full Screen" : "Full Screen";
      button.setAttribute("aria-label", button.title);
      button.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" `
        + `stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${on ? icons.leave : icons.enter}"/></svg>`;
    };
    const enter = () => document.documentElement.requestFullscreen().catch(() => { /* refused: stay windowed */ });
    button.onclick = (event) => {
      event.stopPropagation();
      if (document.fullscreenElement) { remember(false); document.exitFullscreen(); } else { remember(true); enter(); }
    };
    // Leaving with Esc ends it for the session too; leaving because the page changes does not.
    let leaving = false;
    addEventListener("pagehide", () => { leaving = true; });
    addEventListener("beforeunload", () => { leaving = true; });
    document.addEventListener("fullscreenchange", () => { if (!document.fullscreenElement && !leaving) remember(false); show(); });
    if (recall()) {
      const resume = () => { removeEventListener("click", resume, true); removeEventListener("keydown", resume, true);
                             if (recall() && !document.fullscreenElement) enter(); };
      addEventListener("click", resume, true);
      addEventListener("keydown", resume, true);
    }
    show();
    return button;
  }

  // The showcase: the story of the demo as a deck of slides, for presenting.
  function showcaseButton() {
    const link = make("a", { className: "header-button", href: "/showcase", title: "Showcase" });
    link.setAttribute("aria-label", "Open the showcase");
    link.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
      + 'stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="12" rx="2"/><path d="M12 16v4M8 20h8M10 8l4 2-4 2z"/></svg>';
    return link;
  }

  // Dark mode, on or off (theme.js): a moon while it is off, a sun while it is on.
  function themeButton() {
    if (!window.theme) return "";
    const icons = {
      moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
      sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    };
    const button = make("button", { className: "header-button", type: "button" });
    const show = () => {
      const dark = window.theme.current() === "dark";
      button.title = dark ? "Dark Mode Is On: Turn It Off" : "Dark Mode Is Off: Turn It On";
      button.setAttribute("aria-label", "Dark mode");
      button.setAttribute("aria-pressed", String(dark));
      button.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" `
        + `stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${dark ? icons.sun : icons.moon}</svg>`;
    };
    button.onclick = () => window.theme.toggle();
    addEventListener("themechange", show);
    show();
    return button;
  }

  // Settings: the model manager, in a window over the page (settings.js).
  function settingsButton() {
    const button = make("button", { className: "header-button", type: "button", title: "Settings" });
    button.setAttribute("aria-label", "Open settings");
    button.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
      + 'stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1'
      + 'a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1'
      + 'a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1'
      + 'a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1'
      + 'a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>';
    button.onclick = () => window.openSettings?.();
    return button;
  }
  document.head.append(make("script", { src: "/static/settings.js" }));

  // Palette Neat, with its mark: how the pages name it wherever they mention it.
  const neatName = (words) => make("strong", { className: "neat-name" }, make("img", { src: "/static/brand/neat-mark.png", alt: "" }), words);

  const header = document.querySelector("header");
  const title = header.querySelector("h1"), tagline = header.querySelector("p"), nav = header.querySelector("nav");

  // The tabs are written here, so a new page is added in one place. The model manager is not
  // one of them: it is Settings, behind the gear.
  const TABS = [["Debate", "/debate"], ["Questions", "/questions"], ["Vision", "/vision"], ["Compare", "/compare"], ["Games", "/games"]];
  nav.replaceChildren(...TABS.map(([name, href]) => make("a", { href, textContent: name,
    className: location.pathname === href || location.pathname.startsWith(href + "/") ? "here" : "" })));

  const pill = make("span", { className: "header-pill", title: "Everything on this page runs on the board" }, make("i"), "Modalix MLSoC · on-device");
  header.className = "app-header";
  header.replaceChildren(
    make("a", { className: "brand", href: "/" },
      make("img", { src: "/static/brand/neat-decision.svg", alt: "Neat Decision Studio" }),
      make("span", { className: "brand-text" },
        make("span", { className: "brand-title", textContent: "Neat Decision Studio" }),
        make("span", { className: "brand-sub" }, "Running on ", neatName("SiMa.ai Palette Neat")))),
    nav, make("span", { className: "header-tools" }, pill, themeButton(), settingsButton(), showcaseButton(), fullscreenButton()));

  // A page's own title goes below the bar. The landing page has its own opening instead.
  if (document.body.dataset.home === undefined) header.after(make("div", { className: "pagehead" }, title, tagline));

  // The games, in the order the switcher lists them; the first is the one the Games tab opens.
  const GAMES = [["Snake", "/games/snake"], ["Chess", "/games/chess"], ["Tic-Tac-Toe", "/games/tictactoe"],
                 ["Rock Paper Scissors", "/games/rps"], ["Dino Arena", "/games/dino"], ["Blackjack", "/games/blackjack"],
                 ["Sudoku", "/games/sudoku"]];

  // Rows of tiles hold the same number each. A group's columns are the largest number that
  // both fits its width and divides its tile count, so six tiles are 6, 3 + 3 or 2 + 2 + 2 and
  // never 5 + 1. Tiles come and go (a person joins a game) and windows change size, so it is
  // worked out again whenever either happens.
  function evenRows() {
    for (const grid of document.querySelectorAll(".scores, .stats, .metrics, .tiles, .games")) {
      const count = [...grid.children].filter((tile) => !tile.hidden).length;
      if (!count || !grid.clientWidth) continue;
      const least = Number(grid.dataset.min) || (grid.matches(".tiles, .games") ? 215 : grid.matches(".stats") ? 190 : 130);
      const fit = Math.max(1, Math.floor((grid.clientWidth + 8) / (least + 8)));
      let columns = 1;
      for (let k = Math.min(count, fit); k > 1; k--) if (count % k === 0) { columns = k; break; }
      const wanted = `repeat(${columns}, minmax(0, 1fr))`;
      if (grid.style.gridTemplateColumns !== wanted) grid.style.gridTemplateColumns = wanted;
    }
  }

  // The footer goes after everything, so it waits for the rest of the page to be parsed.
  document.addEventListener("DOMContentLoaded", () => {
    evenRows();
    addEventListener("resize", evenRows);
    new MutationObserver(evenRows).observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ["hidden"] });
    // A game page's switcher: the links are written here, so a new game is added in one place.
    const bar = document.querySelector(".gamebar"), here = location.pathname === "/games" ? GAMES[0][1] : location.pathname;
    if (bar && bar.textContent.trim().startsWith("Games")) {
      for (const node of [...bar.childNodes]) if (node.id !== "picker") node.remove();
      bar.prepend("Games ", ...GAMES.map(([name, href]) => make("a", { href, textContent: name, className: href === here ? "here" : "" })));
    }
    document.body.append(make("footer", { className: "app-footer" },
      make("span", {}, "Powered by ", neatName("SiMa.ai Palette Neat"), " on the Modalix MLSoC"),
      make("img", { className: "on-light", src: "/static/brand/sima-on-light.png", alt: "SiMa.ai" }),
      make("img", { className: "on-dark", src: "/static/brand/sima-on-dark.png", alt: "SiMa.ai" }),
      make("span", { textContent: `\u00a9 ${new Date().getFullYear()} SiMa.ai, Inc.` }),
      make("span", { textContent: "Laya models by Convai Innovations, Apache-2.0" })));
  });
})();
