export const REPORT_TARGET_TABS_CSS = `
    .report-target-tabs{display:flex;gap:8px;align-items:center;overflow-x:auto;margin:14px 0 6px;padding:4px;border:1px solid var(--line);border-radius:12px;background:var(--surface-2);scrollbar-width:thin}
    .report-target-tab{flex:0 0 auto;border:1px solid transparent;border-radius:9px;padding:8px 13px;background:transparent;color:var(--muted);font-weight:800;cursor:pointer;white-space:nowrap}
    .report-target-tab:hover{background:var(--surface);color:var(--text)}
    .report-target-tab[aria-selected="true"]{border-color:var(--line);background:var(--surface);color:var(--primary);box-shadow:0 3px 9px rgba(25,39,68,.08)}
    .report-target-panel[hidden]{display:none!important}
    @media(max-width:620px){.report-target-tabs{margin-top:10px}.report-target-tab{padding:7px 10px;font-size:11px}}
    @media print{.report-target-tabs{display:none!important}.report-target-panel[hidden]{display:block!important}}
`;

export function reportTargetTabsRuntimeScript() {
  return `
(() => {
  const sectionDefinitions = [
    { selector: "#comparison", cardSelector: ".comparison-card" },
    { selector: "#diagnostics", cardSelector: ".subcard" },
    { selector: "#model-figures", cardSelector: ".export-target-card" },
  ];
  const groups = [];

  function cardTarget(card) {
    return card.querySelector(".card-heading h3")?.textContent?.trim() || "";
  }

  function resizeVisiblePlots() {
    if (!window.Plotly?.Plots?.resize) return;
    window.requestAnimationFrame(() => {
      document
        .querySelectorAll(".report-target-panel:not([hidden]) .js-plotly-plot")
        .forEach((plot) => window.Plotly.Plots.resize(plot));
    });
  }

  function activateTarget(target) {
    groups.forEach((group) => {
      if (!group.cards.some((item) => item.target === target)) return;
      group.cards.forEach((item) => {
        const active = item.target === target;
        item.card.hidden = !active;
        item.card.setAttribute("aria-hidden", active ? "false" : "true");
      });
      group.buttons.forEach((item) => {
        const active = item.target === target;
        item.button.setAttribute("aria-selected", active ? "true" : "false");
        item.button.tabIndex = active ? 0 : -1;
      });
    });
    resizeVisiblePlots();
  }

  sectionDefinitions.forEach((definition, groupIndex) => {
    const section = document.querySelector(definition.selector);
    if (!section) return;
    const cards = Array.from(section.querySelectorAll(definition.cardSelector))
      .map((card) => ({ card, target: cardTarget(card) }))
      .filter((item) => item.target);
    if (cards.length <= 1) return;

    const tabList = document.createElement("div");
    tabList.className = "report-target-tabs";
    tabList.setAttribute("role", "tablist");
    tabList.setAttribute("aria-label", "目的変数");

    const buttons = cards.map((item, itemIndex) => {
      const panelId = `malchan-target-panel-${groupIndex}-${itemIndex}`;
      const tabId = `malchan-target-tab-${groupIndex}-${itemIndex}`;
      item.card.id = item.card.id || panelId;
      item.card.classList.add("report-target-panel");
      item.card.dataset.reportTarget = item.target;
      item.card.setAttribute("role", "tabpanel");
      item.card.setAttribute("aria-labelledby", tabId);

      const button = document.createElement("button");
      button.type = "button";
      button.id = tabId;
      button.className = "report-target-tab";
      button.dataset.reportTarget = item.target;
      button.setAttribute("role", "tab");
      button.setAttribute("aria-controls", item.card.id);
      button.textContent = item.target;
      button.addEventListener("click", () => activateTarget(item.target));
      tabList.appendChild(button);
      return { button, target: item.target };
    });

    buttons.forEach((item, itemIndex) => {
      item.button.addEventListener("keydown", (event) => {
        let nextIndex = null;
        if (event.key === "ArrowRight") nextIndex = (itemIndex + 1) % buttons.length;
        if (event.key === "ArrowLeft") nextIndex = (itemIndex - 1 + buttons.length) % buttons.length;
        if (event.key === "Home") nextIndex = 0;
        if (event.key === "End") nextIndex = buttons.length - 1;
        if (nextIndex === null) return;
        event.preventDefault();
        buttons[nextIndex].button.focus();
        activateTarget(buttons[nextIndex].target);
      });
    });

    cards[0].card.parentNode.insertBefore(tabList, cards[0].card);
    groups.push({ cards, buttons });
  });

  if (groups.length) activateTarget(groups[0].cards[0].target);
})();`;
}
