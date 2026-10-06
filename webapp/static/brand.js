// The brand header and footer every page shares, in the style of SiMa.ai's Neat demos.
//
// A page writes a plain <header> holding its title (h1), a tagline (p) and the tab links (nav).
// This rebuilds it as the Neat app bar (the mark, the product line and the tabs), moves the
// page's own title below it, and adds the footer. The landing page gets a statement of what
// the demo shows in place of a plain title; every number in it is measured (see the README).
"use strict";
(() => {
  const make = (tag, props = {}, ...children) => {
    const node = Object.assign(document.createElement(tag), props);
    node.append(...children);
    return node;
  };
  const header = document.querySelector("header");
  const title = header.querySelector("h1"), tagline = header.querySelector("p"), nav = header.querySelector("nav");

  const pill = make("span", { className: "header-pill", title: "Everything on this page runs on the board" }, make("i"), "Modalix MLSoC · on-device");
  header.className = "app-header";
  header.replaceChildren(
    make("a", { className: "brand", href: "/" },
      make("img", { src: "/static/brand/neat-mark.png", alt: "Neat" }),
      make("span", { className: "brand-text" },
        make("span", { className: "brand-title", textContent: "Laya Decision Studio" }),
        make("span", { className: "brand-sub" }, "Running on ", make("strong", { textContent: "SiMa.ai Palette Neat" })))),
    nav, pill);

  // The landing page states what the demo shows, in place of a plain title.
  if (document.body.dataset.landing !== undefined) {
    header.after(make("div", { className: "hero-strip" },
      make("div", {},
        make("h1", { textContent: "Decisions at reflex speed, at the edge." }),
        make("p", { textContent: "Ask a yes-or-no question and a 421-million-parameter decision model answers as you type, in about " +
                                 "19 ms, entirely on a SiMa.ai Modalix MLSoC. No cloud, no GPU." })),
      make("div", { className: "hero-facts" },
        make("span", {}, make("b", { textContent: "~19 ms" }), " per decision"),
        make("span", {}, make("b", { textContent: "~45" }), " decisions a second"),
        make("span", {}, make("b", { textContent: "1" }), " MLA graph, ", make("b", { textContent: "0" }), " CPU layers"),
        make("span", {}, make("b", { textContent: "100%" }), " on-device"))));
  } else {
    header.after(make("div", { className: "pagehead" }, title, tagline));
  }

  // The footer goes after everything, so it waits for the rest of the page to be parsed.
  document.addEventListener("DOMContentLoaded", () => {
    document.body.append(make("footer", { className: "app-footer" },
      make("span", {}, "Powered by ", make("strong", { textContent: "SiMa.ai Palette Neat" }), " on the Modalix MLSoC"),
      make("img", { className: "on-light", src: "/static/brand/sima-on-light.png", alt: "SiMa.ai" }),
      make("img", { className: "on-dark", src: "/static/brand/sima-on-dark.png", alt: "SiMa.ai" }),
      make("span", { textContent: `\u00a9 ${new Date().getFullYear()} SiMa.ai, Inc.` }),
      make("span", { textContent: "Laya models by Convai Innovations, Apache-2.0" })));
  });
})();
