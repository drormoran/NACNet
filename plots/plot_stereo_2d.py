from plotly.express import colors as plotly_colors
from plotly import graph_objects as go
from plotly import offline
import numpy as np
import os
from Geometry import stereo_2d
import kornia
import torch
import cv2
from plotly.subplots import make_subplots


def plot_essential_mat_transformation(batch, b_pred_E, b_pred_outliers, base_path, corrected_pts=None, err=None, order=None, n_plots=10, img_loader=None):
    if order is not None:
        plot_indices = order[:n_plots].numpy().tolist()
    else:
        plot_indices = np.arange(b_pred_E.shape[0])[:n_plots].tolist()

    b_gt_shape, b_sampled_match, b_gt_outliers, supp_data = batch
    for i in plot_indices:
        pts1, pts2 = b_sampled_match[i].cpu().split(2, dim=-1)
        pred_T = stereo_2d.shape_to_E(b_pred_E[i].cpu().numpy())
        gt_T = stereo_2d.shape_to_E(b_gt_shape[i].cpu().numpy())
        pred_outliers = b_pred_outliers[i].cpu().numpy()
        gt_outliers = b_gt_outliers[i].cpu().numpy()
        err_i = err[i] if err is not None else None

        # Plot noise distribution
        if corrected_pts is not None:
            crct_pts1, crct_pts2 = corrected_pts[i].cpu().split(2, dim=-1)
            crct_pts1, crct_pts2 = crct_pts1.numpy(), crct_pts2.numpy()
        else:
            crct_pts1, crct_pts2 = None, None
        noise_dist_path = os.path.join(base_path, f'val_set_noise_dist_{supp_data["idx"][i]}.html')
        plot_noise_dist(pts1.numpy(), pts2.numpy(), gt_outliers, gt_T, noise_dist_path, crct_pts1, crct_pts2)

        # Plot epipolar lines
        epi_lines_path = os.path.join(base_path, f'val_set_epi_lines_{supp_data["idx"][i]}.html')
        plot_epipolar_lines(pts1.numpy(), pts2.numpy(), gt_outliers, gt_T, epi_lines_path, crct_pts1, crct_pts2)

        # get images
        if img_loader is not None:
            img1, img2, K1, K2 = img_loader.load(supp_data["idx"][i].cpu().item())
            if img1 is not None:
                pts1 = kornia.geometry.linalg.transform_points(torch.from_numpy(K1).unsqueeze(0).float(), pts1.unsqueeze(0)).squeeze(0)
                pts2 = kornia.geometry.linalg.transform_points(torch.from_numpy(K2).unsqueeze(0).float(), pts2.unsqueeze(0)).squeeze(0)
                pred_T = np.linalg.inv(K2).T @ pred_T @ np.linalg.inv(K1)
        else:
            img1, img2, K1, K2 = None, None, None, None
        pts1, pts2 = pts1.numpy(), pts2.numpy()

        # Plot classification
        class_fig_path = os.path.join(base_path, f'val_set_class_{supp_data["idx"][i]}.html')
        plot_classification(pts1, pts2, pred_outliers, gt_outliers, err_i, class_fig_path, img1, img2)

        # Plot regression
        reg_fig_path = os.path.join(base_path, f'val_set_reg_{supp_data["idx"][i]}.html')
        plot_regression(pts1, pts2, pred_outliers, pred_T, err_i, reg_fig_path, img1, img2)


def plot_epipolar_lines(pts1, pts2, gt_outliers, gt_T, fig_path, crct_pts1=None, crct_pts2=None):
    gt_inliers = (1 - gt_outliers).squeeze(-1).astype(bool)
    if gt_inliers.any():
        pts1 = pts1[gt_inliers]
        pts2 = pts2[gt_inliers]

        if crct_pts1 is not None:
            crct_pts1 = crct_pts1[gt_inliers]
            crct_pts2 = crct_pts2[gt_inliers]

        # GT Correct pts
        gt_crct_pts1, gt_crct_pts2 = stereo_2d.correct_matches(gt_T, pts1, pts2)
        lines_in1 = cv2.computeCorrespondEpilines(gt_crct_pts2.reshape(-1, 1, 2), 2, gt_T).reshape(-1, 3)
        lines_in2 = cv2.computeCorrespondEpilines(gt_crct_pts1.reshape(-1, 1, 2), 1, gt_T).reshape(-1, 3)

        # Plot pts
        pts_idx = np.arange(pts1.shape[0]).astype("str")
        fig_traces = get_stereo_trace(pts1, pts2, pts_idx, gt_crct_pts1, gt_crct_pts2)
        if crct_pts1 is not None:
            fig_traces[0, 0].append(plot_2d_pts(crct_pts1, name='Pred Corrected Pts1', color=plotly_colors.qualitative.D3[1], hoverinfo="x+y+text", hovertext=pts_idx))
        if crct_pts2 is not None:
            fig_traces[0, 1].append(plot_2d_pts(crct_pts2, name='Pred Corrected Pts2', color=plotly_colors.qualitative.D3[1], hoverinfo="x+y+text", hovertext=pts_idx))

        # Plot epipolar lines
        fig_traces[0, 0] += plot_epi_line(lines_in1, np.abs(pts1).max(), pts_idx)
        fig_traces[0, 1] += plot_epi_line(lines_in2, np.abs(pts2).max(), pts_idx)

        fig_traces[0, 0].reverse()
        fig_traces[0, 1].reverse()

        plot_sub_figures(fig_traces, fig_path)


def plot_epi_line(lines, img_bounds, line_idx):
    # B, 2, 2
    lines_pts = np.empty((lines.shape[0], 2, 2))
    lines_pts[:, 0, 0] = -img_bounds
    lines_pts[:, 0, 1] = -(lines[:, 2] - lines[:, 0] * img_bounds) / lines[:, 1]
    lines_pts[:, 1, 0] = img_bounds
    lines_pts[:, 1, 1] = -(lines[:, 2] + lines[:, 0] * img_bounds) / lines[:, 1]

    line_traces = []
    for i, ln_pt in zip(line_idx, lines_pts):
        line_traces.append(plot_2d_pts(ln_pt, mode='lines', name=f'Epipolar Line {i}', legendgroup="epi_lines", legendgrouptitle_text="Epi Lines",
                                               color=plotly_colors.qualitative.D3[7], hoverinfo="x+y+text", hovertext=i, line=dict(width=1)))
    return line_traces


def plot_noise_dist(pts1, pts2, gt_outliers, gt_T, fig_path, crct_pts1=None, crct_pts2=None):
    rep1, rep2, rep_err = calc_rep_err(pts1, pts2, gt_outliers, gt_T)
    go_fig = go.Figure()
    go_fig.add_trace(get_hist_trace(rep1, f"Noise Img1"))
    go_fig.add_trace(get_hist_trace(rep2, f"Noise Img2"))
    go_fig.add_trace(get_hist_trace(rep_err, f"Noise Err"))

    if crct_pts1 is not None:
        _, _, rep_err_crct = calc_rep_err(crct_pts1, crct_pts2, gt_outliers, gt_T)
        go_fig.add_trace(get_hist_trace(rep_err_crct, f"Noise Err (Crct)"))

    go_fig.update_xaxes(title_text="log10(Noise)")
    go_fig.update_layout(showlegend=True)
    offline.plot(go_fig, filename=fig_path, auto_open=False)


def calc_rep_err(pts1, pts2, gt_outliers, gt_T):
    # Remove predicted outliers
    gt_inliers = (1 - gt_outliers).squeeze(-1).astype(bool)
    pts1 = pts1[gt_inliers]
    pts2 = pts2[gt_inliers]

    # Correct pts
    crct_pts1, crct_pts2 = stereo_2d.correct_matches(gt_T, pts1, pts2)
    rep1 = np.linalg.norm(crct_pts1 - pts1, axis=-1, ord=2)
    rep2 = np.linalg.norm(crct_pts2 - pts2, axis=-1, ord=2)
    rep_err = (rep1 + rep2) / 2
    return rep1, rep2, rep_err


def get_hist_trace(x, name):
    x = np.log10(x + 1e-6)
    xbins=dict(start=-6.25, end=0.25, size=0.5)
    trace = go.Histogram(x=x, name=name, xbins=xbins)
    return trace


def plot_regression(pts1, pts2, pred_outliers, pred_T, err, fig_path, img1=None, img2=None):
    # Remove predicted outliers
    pred_inliers = (1 - pred_outliers).squeeze(-1).astype(bool)
    pts1 = pts1[pred_inliers]
    pts2 = pts2[pred_inliers]

    # Correct pts
    if pred_inliers.sum() > 0:
        crct_pts1, crct_pts2 = stereo_2d.correct_matches(pred_T, pts1, pts2)
    else:
        crct_pts1, crct_pts2 = np.empty((0, 2)), np.empty((0, 2))

    # Plot
    pts_idx = np.arange(pts1.shape[0]).astype("str")
    fig_traces = get_stereo_trace(pts1, pts2, pts_idx, crct_pts1, crct_pts2, img1, img2)

    title = f""
    if err is not None:
        title += f", Err: {err:.2f}"

    plot_sub_figures(fig_traces, fig_path, title=title)


def get_stereo_trace(pts1, pts2, pts_idx, crct_pts1=None, crct_pts2=None, img1=None, img2=None):
    fig_traces = np.empty((1, 2), dtype=object)

    fig_traces[0, 0] = []
    fig_traces[0, 0].append(plot_2d_pts(pts1, name='Orig Pts1', color=plotly_colors.qualitative.D3[0],
                                                hoverinfo="x+y+text", hovertext=pts_idx))
    if crct_pts1 is not None:
        fig_traces[0, 0].append(plot_2d_pts(crct_pts1, name='Corrected Pts1', color=plotly_colors.qualitative.D3[2],
                                                    hoverinfo="x+y+text", hovertext=pts_idx))
    if img1 is not None:
        fig_traces[0, 0].append(go.Image(z=img1, hoverinfo="skip"))

    fig_traces[0, 1] = []
    fig_traces[0, 1].append(plot_2d_pts(pts2, name='Orig Pts2', color=plotly_colors.qualitative.D3[0],
                                                hoverinfo="x+y+text", hovertext=pts_idx))
    if crct_pts2 is not None:
        fig_traces[0, 1].append(plot_2d_pts(crct_pts2, name='Corrected Pts2', color=plotly_colors.qualitative.D3[2],
                                                    hoverinfo="x+y+text", hovertext=pts_idx))
    if img2 is not None:
        fig_traces[0, 1].append(go.Image(z=img2, hoverinfo="skip"))

    return fig_traces


def plot_classification(pts1, pts2, pred_outliers, gt_outliers, err, fig_path, img1=None, img2=None):
    # Get color
    incorrect_pred = (pred_outliers != gt_outliers).squeeze(-1)
    correct_pred = (pred_outliers == gt_outliers).squeeze(-1)
    color = np.array([plotly_colors.qualitative.D3[2]] * correct_pred.shape[0]).astype("<U7")
    color[incorrect_pred] = plotly_colors.qualitative.D3[3]

    # Plot fig
    fig_traces = np.empty((1, 2), dtype=object)
    fig_traces[0, 0] = prepare_subfigure(pts1, color, gt_outliers)
    if img1 is not None:
        fig_traces[0, 0].append(go.Image(z=img1, hoverinfo="skip"))

    fig_traces[0, 1] = prepare_subfigure(pts2, color, gt_outliers)
    if img2 is not None:
        fig_traces[0, 1].append(go.Image(z=img2, hoverinfo="skip"))

    # Set title
    title = f""
    if err is not None:
        title += f", Err: {err:.2f}"

    plot_sub_figures(fig_traces, fig_path, title=title)


def plot_2d_pts(pts, mode='markers', name=None, color=None, size=None, **kwargs):
    return go.Scatter(x=pts[..., 0], y=pts[..., 1], mode=mode, name=name, marker=dict(color=color, size=size), **kwargs)


def prepare_subfigure(pts, colors, gt_outliers, img=None):
    data = []
    pts_idx = np.arange(pts.shape[0]).astype("str")

    # Plot image
    if img is not None:
        data.append(go.Image(z=kornia.utils.tensor_to_image(img * 255)))

    # Plot outliers
    outliers_mask = (gt_outliers == 1).squeeze()
    data.append(plot_2d_pts(pts[outliers_mask], name='GT Outliers', color=colors[outliers_mask],
                                    hoverinfo="x+y+text", hovertext=pts_idx[outliers_mask]))

    # Plot inliers
    inliers_mask = (gt_outliers == 0).squeeze()
    data.append(plot_2d_pts(pts[inliers_mask], name='GT Intliers', color=colors[inliers_mask],
                                    hoverinfo="x+y+text", hovertext=pts_idx[inliers_mask]))
    return data


def plot_sub_figures(fig_traces, path, title=""):
    go_fig = make_subplots(rows=fig_traces.shape[0], cols=fig_traces.shape[1])
    # Plot traces
    for i in range(fig_traces.shape[0]):
        for j in range(fig_traces.shape[1]):
            for trace in fig_traces[i, j]:
                go_fig.add_trace(trace, row=i + 1, col=j + 1)

    # Update axes
    for i in range(fig_traces.shape[0]):
        for j in range(fig_traces.shape[1]):
            sub_figure_idx = i * fig_traces.shape[1] + j + 1
            go_fig.update_yaxes(scaleanchor=f"x{sub_figure_idx}", scaleratio=1, row=i + 1, col=j + 1)

    go_fig.update_layout(title=title)
    offline.plot(go_fig, filename=path, auto_open=False)