"""Plot direct VBM, VBM after conversion through BAGLE, and Gulls truth.

Example (with the two downloaded Beginner answer Parquets)::

    python examples/gulls_round_trip.py truth.parquet curves.parquet \
        --event RMDC26_000093 --output comparison.html

The figure opens around the F146 maximum. Both model lines use a finer time
grid than the observations. No parameters are fitted. Plotly is needed only
for this figure, not for the conversion or its regression test.
"""
import argparse
from pathlib import Path

import numpy as np

from bagle import frame_convert
from bagle.gulls import JD_ZERO, JD_TO_MJD, load_event


def comparison(truth_path, curve_path, name):
    """Evaluate both fixed models around the F146 maximum of one event.

    Parameters
    ----------
    truth_path, curve_path : path-like
        Beginner answer Parquets.
    name : str
        RMDC26 event name.

    Returns
    -------
    result : dict
        Observation times/fluxes, model grid/fluxes, and the peak epoch.
    """
    from importlib.metadata import version
    event = load_event(name, truth_path, curve_path, columns=[
        'Simulation_time', 'observatory_code', 'true_relative_flux'])
    data = event.curve.to_pandas()
    data = data[data.observatory_code == 0].sort_values('Simulation_time')
    peak = float(data.loc[data.true_relative_flux.idxmax(), 'Simulation_time'])
    selected = data[abs(data.Simulation_time-peak) <= .6]
    time = selected.Simulation_time.to_numpy()
    truth = selected.true_relative_flux.to_numpy()
    # A 4097-point grid gives >20 model samples per typical native cadence.
    grid = np.unique(np.r_[np.linspace(time.min(), time.max(), 4097), time])
    direct = event.predict(grid, band=0, accuracy=1e-4, relative_accuracy=1e-6)
    model = event.to_bagle_model()
    del model.raw
    mjd = (JD_ZERO-JD_TO_MJD)+grid
    vbm, parameters = frame_convert.get_vbm_model(model, times_mjd=mjd)
    # Both VBM paths use BinaryMag2, Tol=1e-4, RelTol=1e-6,
    # minannuli=1 and a1=0 (uniform brightness), with no light-time correction.
    routine = (vbm.BinSourceBinLensLightCurve if len(parameters) == 18
               else vbm.BinaryLightCurveParallax)
    result = np.asarray(routine(parameters, (mjd-49999.5).tolist()))
    converted = 1 + model.b_sff[0]*(result[0]-1)
    return dict(name=name, time=time, truth=truth, grid=grid, direct=direct,
                converted=converted, index=np.searchsorted(grid, time), peak=peak,
                vbm_version=version('VBMicrolensing'))


def figure(result):
    """Return an interactive figure and its standalone scientific caption."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    d = result
    x = d['time']-d['peak']
    xx = d['grid']-d['peak']
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[.55, .28, .17], vertical_spacing=.035)
    colors = ['#d97706', '#0284c7']
    names = ['Independent Gulls to VBM',
             'BAGLE ingesting Gulls and exporting VBM']
    for flux, color, label, dash in zip(
            [d['direct'], d['converted']], colors, names, ['solid', 'dash']):
        fig.add_trace(go.Scatter(x=xx, y=flux, name=label, legendgroup=label,
                                line=dict(color=color, width=2, dash=dash)),
                      row=1, col=1)
        residual = (flux[d['index']]/d['truth']-1)*1e6
        fig.add_trace(go.Scatter(x=x, y=residual, name=label, legendgroup=label,
                                showlegend=False, mode='lines+markers',
                                marker=dict(size=4, color=color),
                                line=dict(color=color, width=1, dash=dash)),
                      row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=d['truth'], name='Gulls Noiseless',
                            mode='markers', marker=dict(size=5, color='#475569',
                                                       symbol='circle-open')),
                  row=1, col=1)
    delta = (d['converted'][d['index']]/d['direct'][d['index']]-1)*1e6
    fig.add_trace(go.Scatter(x=x, y=delta, name='BAGLE-exported VBM relative to independent VBM',
                            showlegend=False, mode='lines+markers',
                            line=dict(color='#a855f7', width=1),
                            marker=dict(size=4)), row=3, col=1)
    fig.update_yaxes(title_text='Relative flux', row=1, col=1)
    fig.update_yaxes(title_text='Residual<br>vs Gulls (ppm)', row=2, col=1)
    fig.update_yaxes(title_text='Residual vs<br>independent VBM (ppm)', row=3, col=1)
    fig.update_xaxes(title_text=f'Simulation time − {d["peak"]:.9f} (days)',
                     row=3, col=1)
    fig.update_xaxes(range=[-.12, .12])
    # Each button explicitly resets all linked panels to the same time range.
    buttons = []
    for width, label in [(.12, 'Peak ±0.12 d'), (.6, '±0.6 d')]:
        update = {f'xaxis{i if i > 1 else ""}.range': [-width, width]
                  for i in range(1, 4)}
        buttons.append(dict(label=label, method='relayout', args=[update]))
    fig.update_layout(
        height=790, margin=dict(l=96, r=18, t=185, b=75),
        title=dict(text='RMDC26 Beginner Tier Parameterization Conversion Comparisons'
                   f'<br><sup>{d["name"]} · F146 · fixed truth</sup>',
                   x=0., xanchor='left', y=.995, yanchor='top',
                   font=dict(size=15)),
        legend=dict(orientation='v', x=0, y=1.18, yanchor='top',
                    font=dict(size=11)),
        updatemenus=[dict(type='buttons', direction='right', x=0., y=-.12,
                          buttons=buttons)],
        template='plotly_white', hovermode='x unified', dragmode='pan')
    caption = (
        'This figure validates BAGLE’s conversion of the RMDC26 Beginner Tier '
        'Gulls microlensing model to VBM against our best-matching independent '
        f'direct conversion, using {d["name"]} in F146. '
        'Orange is “Independent Gulls to VBM”, dashed blue is “BAGLE ingesting '
        'Gulls and exporting VBM”, and open circles are “Gulls Noiseless”. '
        'The middle panel, “vs Gulls”, uses Gulls Noiseless as the reference: '
        '(model flux / Gulls Noiseless flux − 1) × 10⁶ for each model. '
        'The bottom panel uses Independent Gulls to VBM as the reference: '
        '(BAGLE-exported VBM flux / independent VBM flux − 1) × 10⁶. '
        'Both residuals are evaluated at observation times, and ppm means '
        'parts per million of the reference flux. Positive values mean the '
        'numerator model is brighter than its reference. Both model lines '
        'use a finer time grid than the observations. Both calculations use '
        f'VBMicrolensing {d["vbm_version"]}, BinaryMag2, Tol=1e-4, RelTol=1e-6, '
        'minannuli=1 and a1=0 (uniform brightness). Times use Simulation_time '
        'since JD 2461406.5. No parameters were fitted. BAGLE supplies the '
        'parameter conversion and VBM supplies both finite-source calculations.'
    )
    return fig, caption


def main():
    import html
    import plotly.io as pio
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('truth', type=Path)
    parser.add_argument('curves', type=Path)
    parser.add_argument('--event', default='RMDC26_000093')
    parser.add_argument('--output', type=Path, default=Path('gulls_round_trip.html'))
    args = parser.parse_args()
    fig, caption = figure(comparison(args.truth, args.curves, args.event))
    content = pio.to_html(fig, include_plotlyjs=True, full_html=False,
                          config=dict(scrollZoom=True, displaylogo=False))
    args.output.write_text('<!doctype html><html><head><meta charset="utf-8">'
                           '</head><body>'+content+'<p>'+html.escape(caption)+
                           '</p></body></html>')
    print(args.output)


if __name__ == '__main__':
    main()
