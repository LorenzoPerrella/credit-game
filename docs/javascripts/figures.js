// Figures follow the colour scheme: Plotly draws its text in a fixed colour, so the text is
// set again whenever the reader switches between the light and the dark palette.
(function () {
  function recolour() {
    if (typeof Plotly === "undefined") return;
    var colour = getComputedStyle(document.body).getPropertyValue("--md-typeset-color").trim();
    document.querySelectorAll(".plotly-graph-div").forEach(function (figure) {
      if (figure.layout) Plotly.relayout(figure, { "font.color": colour || "#333333" });
    });
  }
  new MutationObserver(recolour).observe(document.body, {
    attributes: true,
    attributeFilter: ["data-md-color-scheme"],
  });
  window.addEventListener("load", recolour);
})();
