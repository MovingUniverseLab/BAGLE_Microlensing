katex_options = {
macros: {"\\i":               "\\mathrm{i}",
"\\e":             "\\mathrm{e}^{#1}",
"\\vec":           "\\mathbf{#1}",
"\\x":               "\\vec{x}",
"\\d":               "\\operatorname{d}\\!",
"\\dirac":         "\\operatorname{\\delta}\\left(#1\\right)",
"\\scalarprod":  "\\left\\langle#1,#2\\right\\rangle",},
delimiters: [
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true }
        ]
}
document.addEventListener("DOMContentLoaded", function() {
  renderMathInElement(document.body, katex_options);
});
