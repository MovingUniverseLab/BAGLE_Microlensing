"""Self-contained HTML report for PSBL sampler comparisons."""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path


def _png_data_uri(path: str | Path | None) -> str:
    """Embed a PNG file as a data URI, or return an empty placeholder."""
    if path is None:
        return ""
    path = Path(path)
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _fmt_runtime(seconds):
    """Format elapsed seconds for display."""
    try:
        sec = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if not np_isfinite(sec):
        return "—"
    if sec < 60:
        return f"{sec:.1f} s"
    if sec < 3600:
        return f"{sec / 60.0:.1f} min"
    return f"{sec / 3600.0:.2f} hr"


def np_isfinite(x):
    """Finite check without importing numpy at module import time."""
    try:
        return x == x and abs(x) != float("inf")
    except Exception:
        return False


def _fmt_float(x, prec=3):
    """Format a float, tolerating missing values."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np_isfinite(xf):
        if xf == float("inf"):
            return "∞"
        if xf == float("-inf"):
            return "-∞"
        return "—"
    return f"{xf:.{prec}f}"


def _bayes_factor_rows(results):
    """Build ΔlogZ / Bayes-factor rows among nested-sampling runs."""
    import math

    nested = [
        r for r in results
        if r.get("status") == "ok" and np_isfinite(r.get("logZ", float("nan")))
    ]
    rows = []
    for i, a in enumerate(nested):
        for b in nested[i + 1 :]:
            dlogz = float(a["logZ"]) - float(b["logZ"])
            # Avoid OverflowError for huge ΔlogZ in early/failed evidence.
            if dlogz > 700:
                bf = float("inf")
            elif dlogz < -700:
                bf = 0.0
            else:
                bf = float(math.exp(dlogz))
            rows.append(
                {
                    "a": a["label"],
                    "b": b["label"],
                    "dlogz": dlogz,
                    "bf_ab": bf,
                }
            )
    return rows


def write_html_report(results, out_path, title=None, subtitle=None):
    """Write a self-contained HTML comparison report.

    Parameters
    ----------
    results : list of dict
        Per-backend result records from ``run_comparison.run_one``.
    out_path : str or Path
        Destination HTML file.
    title : str, optional
        Page / ``<h1>`` title.
    subtitle : str, optional
        Short description under the title.

    Returns
    -------
    out_path : Path
        Path written.
    """
    out_path = Path(out_path)
    ok = [r for r in results if r.get("status") == "ok"]
    if title is None:
        title = "PSBL Phot+Astrom sampler comparison"
    if subtitle is None:
        subtitle = (
            "Fake noisy photometry + astrometry · "
            "MultiNest (JAX lnL) · NumPyro NUTS / SA / SMC-NUTS · "
            "PyMC SMC · jaxns ± gradient-guided"
        )

    # Summary metric cards.
    metric_html = []
    for r in results:
        status = r.get("status", "?")
        runtime = _fmt_runtime(r.get("runtime_sec"))
        lnL = _fmt_float(r.get("jax_lnL"), 2)
        logz = _fmt_float(r.get("logZ"), 2)
        metric_html.append(
            f"""
  <div class="metric">
    <span>{escape(r.get('label', ''))}</span>
    <strong>{escape(status)}</strong>
    <div class="sub">runtime {escape(runtime)} · lnL {escape(lnL)} · logZ {escape(logz)}</div>
  </div>"""
        )

    # Runtime / lnL table.
    summary_rows = []
    for r in results:
        summary_rows.append(
            "<tr>"
            f"<td>{escape(r.get('label', ''))}</td>"
            f"<td>{escape(r.get('status', ''))}</td>"
            f"<td>{escape(_fmt_runtime(r.get('runtime_sec')))}</td>"
            f"<td>{escape(_fmt_float(r.get('jax_lnL'), 3))}</td>"
            f"<td>{escape(_fmt_float(r.get('host_lnL'), 3))}</td>"
            f"<td>{escape(_fmt_float(r.get('logZ'), 3))}</td>"
            f"<td class='err'>{escape(str(r.get('error') or '')[:120])}</td>"
            "</tr>"
        )

    # Parameter table using first successful run's param order.
    param_names = []
    for r in ok:
        param_names = list(r.get("param_names") or [])
        if param_names:
            break

    truth = ok[0]["truth"] if ok else {}
    header = (
        "<tr><th>parameter</th><th>truth</th>"
        + "".join(f"<th>{escape(r['label'])}</th>" for r in ok)
        + "".join(f"<th>bias:{escape(r['label'])}</th>" for r in ok)
        + "</tr>"
    )
    param_rows = []
    for name in param_names:
        cells = [f"<td>{escape(name)}</td>", f"<td>{_fmt_float(truth.get(name), 5)}</td>"]
        for r in ok:
            cells.append(f"<td>{_fmt_float(r['best'].get(name), 5)}</td>")
        for r in ok:
            bias = r.get("bias", {}).get(name)
            cls = "bias-neutral"
            try:
                b = abs(float(bias))
                t = abs(float(truth.get(name, 0.0))) + 1e-12
                rel = b / t
                if rel < 0.01 or b < 1e-3:
                    cls = "bias-good"
                elif rel < 0.1:
                    cls = "bias-warning"
                else:
                    cls = "bias-bad"
            except Exception:
                pass
            cells.append(f"<td class='{cls}'>{_fmt_float(bias, 5)}</td>")
        param_rows.append("<tr>" + "".join(cells) + "</tr>")

    bf_rows = _bayes_factor_rows(results)
    bf_html = []
    for row in bf_rows:
        bf_html.append(
            "<tr>"
            f"<td>{escape(row['a'])}</td>"
            f"<td>{escape(row['b'])}</td>"
            f"<td>{_fmt_float(row['dlogz'], 3)}</td>"
            f"<td>{_fmt_float(row['bf_ab'], 3)}</td>"
            "</tr>"
        )
    if not bf_html:
        bf_html.append(
            "<tr><td colspan='4'>No nested-sampling logZ pairs available yet.</td></tr>"
        )

    # Figure sections.
    figure_sections = []
    for r in ok:
        trace_uri = _png_data_uri(r.get("trace_png"))
        model_uri = _png_data_uri(r.get("model_png"))
        figure_sections.append(
            f"""
<section>
  <h2>{escape(r['label'])} — traces &amp; model/data</h2>
  <div class="grid">
    <div><h3>Trace / sample path</h3>
      {"<img src='" + trace_uri + "' alt='trace'/>" if trace_uri else "<p class='note'>No trace figure.</p>"}
    </div>
    <div><h3>Model vs data</h3>
      {"<img src='" + model_uri + "' alt='model'/>" if model_uri else "<p class='note'>No model figure.</p>"}
    </div>
  </div>
</section>"""
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
body {{
  margin: 0;
  background: #f4f5f7;
  color: #20242a;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.45;
}}
main {{ max-width: 1280px; margin: 0 auto; padding: 30px 24px 60px; }}
h1, h2, h3 {{ color: #172033; }}
h1 {{ margin-bottom: 4px; }}
.subtitle {{ color: #5b6472; margin-top: 0; }}
.metrics {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin: 24px 0;
}}
.metric, section {{
  background: white;
  border: 1px solid #d9dde5;
  border-radius: 7px;
}}
.metric {{ padding: 14px 16px; }}
.metric span {{ display: block; color: #687284; font-size: 0.82rem; }}
.metric strong {{ font-size: 1.15rem; }}
.metric .sub {{ color: #687284; font-size: 0.8rem; margin-top: 6px; }}
section {{ margin-top: 20px; padding: 20px; }}
img {{ display: block; width: 100%; height: auto; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
.table-wrap {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #e3e6ec; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #eef1f5; color: #30394a; }}
.bias-good {{ background: #dff3e4; color: #176b34; font-weight: 600; }}
.bias-warning {{ background: #fff1bf; color: #795b00; font-weight: 600; }}
.bias-bad {{ background: #f8d7da; color: #8a1c25; font-weight: 600; }}
.bias-neutral {{ color: #687284; }}
.note {{ color: #5b6472; font-size: 0.9rem; }}
td.err {{ text-align: left; color: #8a1c25; font-size: 0.8rem; max-width: 240px; overflow: hidden; }}
@media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<main>
<h1>{escape(title)}</h1>
<p class="subtitle">{subtitle}</p>

<div class="metrics">
{''.join(metric_html)}
</div>

<section>
  <h2>Runtimes &amp; likelihoods</h2>
  <div class="table-wrap">
  <table>
    <tr>
      <th>backend</th><th>status</th><th>runtime</th>
      <th>best JAX lnL</th><th>best host lnL</th><th>logZ</th><th>error</th>
    </tr>
    {''.join(summary_rows)}
  </table>
  </div>
  <p class="note">
  Bayes factors use nested-sampling log-evidence only (MultiNest / jaxns).
  NUTS and SA leave logZ blank. Bias columns are best−truth.
  </p>
</section>

<section>
  <h2>Bayes factors (nested sampling)</h2>
  <div class="table-wrap">
  <table>
    <tr><th>A</th><th>B</th><th>ΔlogZ (A−B)</th><th>BF A/B</th></tr>
    {''.join(bf_html)}
  </table>
  </div>
</section>

<section>
  <h2>Best-fit parameters &amp; bias</h2>
  <div class="table-wrap">
  <table>
    {header}
    {''.join(param_rows) if param_rows else '<tr><td colspan="99">No successful fits yet.</td></tr>'}
  </table>
  </div>
</section>

{''.join(figure_sections)}

</main>
</body>
</html>
"""
    out_path.write_text(html)
    return out_path


def write_suite_summary_html(suite_results, out_path, title=None):
    """Write a cross-model suite summary (timing, lnL, logZ, bias).

    Parameters
    ----------
    suite_results : list of dict
        Flattened result records from one or more scenarios / models.
        Each record should include ``label``, ``model``, ``scenario``,
        ``runtime_sec``, ``jax_lnL``, ``logZ``, ``bias``, ``truth``,
        and optional ``jax`` / ``grads`` / ``family`` metadata.
    out_path : str or Path
        Destination HTML file.
    title : str, optional
        Page title.

    Returns
    -------
    out_path : Path
        Path written.
    """
    out_path = Path(out_path)
    if title is None:
        title = "Sampler suite summary — timing, accuracy, bias, logZ"

    # Group by model / scenario for section headings.
    groups = {}
    for r in suite_results:
        key = (r.get("model", "?"), r.get("scenario", "?"))
        groups.setdefault(key, []).append(r)

    sections = []
    for (model, scenario), results in sorted(groups.items()):
        ok = [r for r in results if r.get("status") == "ok"]

        # Timing / accuracy / logZ table.
        summary_rows = []
        for r in results:
            jax_flag = r.get("jax")
            grads_flag = r.get("grads")
            jax_s = "yes" if jax_flag is True else ("no" if jax_flag is False else "—")
            grads_s = (
                "yes" if grads_flag is True
                else ("no" if grads_flag is False else "—")
            )
            summary_rows.append(
                "<tr>"
                f"<td>{escape(r.get('label', ''))}</td>"
                f"<td>{escape(r.get('family', ''))}</td>"
                f"<td>{escape(jax_s)}</td>"
                f"<td>{escape(grads_s)}</td>"
                f"<td>{escape(r.get('status', ''))}</td>"
                f"<td>{escape(_fmt_runtime(r.get('runtime_sec')))}</td>"
                f"<td>{escape(_fmt_float(r.get('jax_lnL'), 3))}</td>"
                f"<td>{escape(_fmt_float(r.get('host_lnL'), 3))}</td>"
                f"<td>{escape(_fmt_float(r.get('logZ'), 3))}</td>"
                f"<td class='err'>{escape(str(r.get('error') or '')[:100])}</td>"
                "</tr>"
            )

        # Bias table.
        param_names = []
        for r in ok:
            param_names = list(r.get("param_names") or [])
            if param_names:
                break
        truth = ok[0].get("truth", {}) if ok else {}
        bias_header = (
            "<tr><th>parameter</th><th>truth</th>"
            + "".join(f"<th>{escape(r['label'])}</th>" for r in ok)
            + "".join(f"<th>bias:{escape(r['label'])}</th>" for r in ok)
            + "</tr>"
        )
        bias_rows = []
        for name in param_names:
            cells = [
                f"<td>{escape(name)}</td>",
                f"<td>{_fmt_float(truth.get(name), 5)}</td>",
            ]
            for r in ok:
                cells.append(
                    f"<td>{_fmt_float(r.get('best', {}).get(name), 5)}</td>"
                )
            for r in ok:
                bias = r.get("bias", {}).get(name)
                cls = "bias-neutral"
                try:
                    b = abs(float(bias))
                    t = abs(float(truth.get(name, 0.0))) + 1e-12
                    rel = b / t
                    if rel < 0.01 or b < 1e-3:
                        cls = "bias-good"
                    elif rel < 0.1:
                        cls = "bias-warning"
                    else:
                        cls = "bias-bad"
                except Exception:
                    pass
                cells.append(f"<td class='{cls}'>{_fmt_float(bias, 5)}</td>")
            bias_rows.append("<tr>" + "".join(cells) + "</tr>")

        # Mean |relative bias| score per backend (accuracy summary).
        acc_rows = []
        for r in ok:
            biases = r.get("bias", {}) or {}
            rels = []
            for name, b in biases.items():
                try:
                    t = abs(float(truth.get(name, 0.0))) + 1e-12
                    rels.append(abs(float(b)) / t)
                except Exception:
                    continue
            mean_rel = float(sum(rels) / len(rels)) if rels else float("nan")
            max_rel = float(max(rels)) if rels else float("nan")
            acc_rows.append(
                "<tr>"
                f"<td>{escape(r['label'])}</td>"
                f"<td>{_fmt_float(mean_rel, 4)}</td>"
                f"<td>{_fmt_float(max_rel, 4)}</td>"
                f"<td>{_fmt_float(r.get('jax_lnL'), 3)}</td>"
                f"<td>{_fmt_float(r.get('logZ'), 3)}</td>"
                f"<td>{_fmt_runtime(r.get('runtime_sec'))}</td>"
                "</tr>"
            )

        sections.append(
            f"""
<section>
  <h2>{escape(str(model).upper())} · {escape(str(scenario))}</h2>
  <p class="note">{escape(str(results[0].get('prior_mode', '')))} priors ·
     {len(ok)}/{len(results)} backends ok</p>

  <h3>Timing, accuracy (lnL), evidence</h3>
  <div class="table-wrap">
  <table>
    <tr><th>backend</th><th>family</th><th>JAX</th><th>grads</th>
        <th>status</th><th>runtime</th><th>jax lnL</th><th>host lnL</th>
        <th>logZ</th><th>error</th></tr>
    {''.join(summary_rows)}
  </table>
  </div>

  <h3>Accuracy summary (bias vs truth)</h3>
  <div class="table-wrap">
  <table>
    <tr><th>backend</th><th>mean |rel bias|</th><th>max |rel bias|</th>
        <th>jax lnL</th><th>logZ</th><th>runtime</th></tr>
    {''.join(acc_rows) if acc_rows else '<tr><td colspan="6">No successful fits.</td></tr>'}
  </table>
  </div>

  <h3>Best-fit parameters &amp; bias</h3>
  <div class="table-wrap">
  <table>
    {bias_header}
    {''.join(bias_rows) if bias_rows else '<tr><td colspan="99">No successful fits.</td></tr>'}
  </table>
  </div>
</section>
"""
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
body {{
  margin: 0; background: #f4f5f7; color: #20242a;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.45;
}}
main {{ max-width: 1400px; margin: 0 auto; padding: 30px 24px 60px; }}
h1, h2, h3 {{ color: #172033; }}
.note {{ color: #5b6472; }}
section {{
  background: white; border: 1px solid #d9dde5; border-radius: 7px;
  margin-top: 20px; padding: 20px;
}}
.table-wrap {{ overflow-x: auto; }}
table {{
  width: 100%; border-collapse: collapse;
  font-variant-numeric: tabular-nums;
}}
th, td {{
  padding: 8px 10px; border-bottom: 1px solid #e3e6ec;
  text-align: right; white-space: nowrap;
}}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #eef1f5; color: #30394a; }}
.bias-good {{ background: #dff3e4; color: #176b34; font-weight: 600; }}
.bias-warning {{ background: #fff1bf; color: #795b00; font-weight: 600; }}
.bias-bad {{ background: #f8d7da; color: #8a1c25; font-weight: 600; }}
.bias-neutral {{ color: #687284; }}
td.err {{ text-align: left; color: #8a1c25; font-size: 0.8rem; max-width: 220px; }}
</style>
</head>
<body>
<main>
<h1>{escape(title)}</h1>
<p class="note">Narrow-prior timing &amp; accuracy across MultiNest
(host/JAX), NumPyro (NUTS / SA / SMC-NUTS), PyMC SMC (JAX / host),
and jaxns (± gradient-guided).</p>
{''.join(sections)}
</main>
</body>
</html>
"""
    out_path.write_text(html)
    return out_path
