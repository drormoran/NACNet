from plotly import graph_objects as go
from plotly import offline
import numpy as np
import os


def plot_samples_err_distribution(all_samples_errors, base_path):
    names = list(all_samples_errors.keys())

    # Get errors names
    error_names = [col for col in all_samples_errors[names[0]].columns if not col.startswith("meta")]

    for err_name in error_names:
        hist_path = os.path.join(base_path, f'Err_dist_{err_name}.html')
        err_values = []
        for method_name in names:
            err_values.append(all_samples_errors[method_name][err_name])
        plot_histogram(err_values, hist_path, names=names, hist_max_range=(0, 20))


def plot_err_vs_meta(samples_errors, base_path):
    meta_cols = [col for col in samples_errors.columns if col.startswith("meta")]
    err_cols = [col for col in samples_errors.columns if not col.startswith("meta")]
    for err_col in err_cols:
        for meta_col in meta_cols:
            xs = [samples_errors[meta_col]]
            ys = [samples_errors[err_col]]
            names = ["Ours"]
            scatter_path = os.path.join(base_path, f'{err_col}_vs_{meta_col}.html')
            plot_scatter(xs, ys, scatter_path, names, meta_col, err_col)


def plot_scatter(xs, ys, res_path, names, xaxis_title, yaxis_title):
    go_fig = go.Figure()
    for x, y, name in zip(xs, ys, names):
        go_fig.add_trace(go.Scatter(x=x, y=y, name=f"{name} Mean:{y.mean():.3f},Std:{y.std():.3f}", mode='markers'))

    go_fig.update_layout(showlegend=True, xaxis_title=xaxis_title, yaxis_title=yaxis_title,)
    offline.plot(go_fig, filename=res_path, auto_open=False)


def plot_histogram(xs, res_path, hist_max_range=(-np.inf, np.inf), names=None):
    if not isinstance(xs, list):
        xs = [xs]

    if names is None:
        names = [""] * len(xs)

    min_val = np.inf
    max_val = -np.inf
    go_fig = go.Figure()
    for x, name in zip(xs, names):
        clipped_x = x.clip(*hist_max_range)
        min_val = min(clipped_x.min(), min_val)
        max_val = max(clipped_x.max(), max_val)
        mAA = calc_mAA(x.to_numpy())
        go_fig.add_trace(go.Histogram(x=clipped_x, name=f"{name} Mean:{x.mean():.3f},Std:{x.std():.3f}, mAA:{mAA:.3f}"))
    go_fig.update_xaxes(range=(min_val - 1, max_val + 1))
    go_fig.update_layout(showlegend=True)
    offline.plot(go_fig, filename=res_path, auto_open=False)


def calc_mAA(MAEs, ths=np.logspace(np.log2(1.0), np.log2(20), 10, base=2.0)):
    acc = np.expand_dims(MAEs, -1) <= np.expand_dims(ths, 0)
    cur_results = acc.mean(axis=-1)
    res = cur_results.mean()
    return res
