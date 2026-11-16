"""Plotly helpers for the interactive Thermal Pattern Propagation Lab UI."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass(frozen=True)
class PreviewVideo:
    frames: np.ndarray
    frame_indices: np.ndarray
    spatial_stride: int


def downsample_video(
    frames: np.ndarray,
    max_frames: int = 96,
    max_side: int = 180,
) -> PreviewVideo:
    """Downsample only the browser preview, leaving the analysis arrays untouched."""
    arr = np.asarray(frames)
    if arr.ndim != 3 or arr.shape[0] < 1:
        raise ValueError("Preview expects a non-empty (frames, height, width) array")
    if max_frames < 2 or max_side < 8:
        raise ValueError("max_frames must be >= 2 and max_side must be >= 8")
    count = min(arr.shape[0], int(max_frames))
    indices = np.unique(np.linspace(0, arr.shape[0] - 1, count).round().astype(int))
    stride = max(1, int(math.ceil(max(arr.shape[1:]) / max_side)))
    preview = arr[indices, ::stride, ::stride].astype(np.float32, copy=False)
    return PreviewVideo(preview, indices, stride)


def robust_color_limits(data: np.ndarray, symmetric: bool = False) -> tuple[float, float]:
    """Robust, fixed color limits for all frames in an animation."""
    values = np.asarray(data, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    if symmetric:
        bound = float(np.nanpercentile(np.abs(finite), 98))
        bound = max(bound, 1e-9)
        return -bound, bound
    low, high = [float(v) for v in np.nanpercentile(finite, [1, 99])]
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        center = float(np.nanmedian(finite))
        return center - 1.0, center + 1.0
    return low, high


def make_frame_player(
    frames: np.ndarray,
    times_s: np.ndarray,
    roi_bounds: tuple[int, int, int, int] | None = None,
    view_mode: str = "Thermal sequence",
    unit_label: str = "°C",
    playback_speed: float = 1.0,
    max_frames: int = 96,
    max_side: int = 180,
    roi_regions: Sequence[dict] | None = None,
    phase_labels: Sequence[str] | None = None,
) -> tuple[go.Figure, dict]:
    """Build a browser-side animated heatmap with play/pause and a time slider.

    ``frames`` is the visualization channel (raw, delta, or derivative). It is
    downsampled for browser performance only; quantitative analysis uses the
    original resolution.
    """
    arr = np.asarray(frames, dtype=np.float32)
    preview = downsample_video(arr, max_frames=max_frames, max_side=max_side)
    video = preview.frames
    indices = preview.frame_indices
    all_times = np.asarray(times_s, dtype=float)
    if all_times.shape != (arr.shape[0],):
        raise ValueError("times_s must contain one timestamp per frame")
    times = all_times[indices]
    phases = None if phase_labels is None else np.asarray(phase_labels, dtype=object)
    if phases is not None and phases.shape != (arr.shape[0],):
        raise ValueError("phase_labels must contain one label per frame")
    h, w = video.shape[1:]
    scale = preview.spatial_stride

    mode_colors = {
        "Thermal sequence": "Inferno",
        "Δ from baseline": "RdBu_r",
        "Rate of change": "Turbo",
        "Event progression": "RdBu_r",
    }
    colorscale = mode_colors.get(view_mode, "Inferno")
    symmetric = view_mode in {"Δ from baseline", "Rate of change", "Event progression"}
    zmin, zmax = robust_color_limits(video, symmetric=symmetric)
    if view_mode == "Rate of change":
        cbar_title = f"dT/dt ({unit_label}/s)"
    elif view_mode == "Event progression":
        cbar_title = f"Δ from event reference ({unit_label})"
    elif view_mode == "Δ from baseline":
        cbar_title = f"Δ ({unit_label})"
    else:
        cbar_title = unit_label

    def title_for(i: int) -> str:
        phase = f"  ·  {phases[indices[i]]}" if phases is not None else ""
        return f"{view_mode}  ·  Frame {int(indices[i]) + 1}/{arr.shape[0]}  ·  {times[i]:.2f} s{phase}"

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=video[0], x=np.arange(w), y=np.arange(h),
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        colorbar={"title": {"text": cbar_title}, "thickness": 14},
        hovertemplate="x=%{x}<br>y=%{y}<br>value=%{z:.3f}<extra></extra>",
        name=view_mode,
    ))
    # ROI outlines are separate, non-animated traces so they remain visible during playback.
    if roi_regions:
        for region in roi_regions:
            points = np.asarray(region.get("points", []), dtype=float)
            if points.ndim == 2 and len(points) >= 3:
                x = (points[:, 0] / scale).tolist()
                y = (points[:, 1] / scale).tolist()
                fig.add_trace(go.Scatter(
                    x=x + [x[0]], y=y + [y[0]], mode="lines",
                    line={"color": region.get("color", "#F8FAFC"), "width": 2.2},
                    hoverinfo="skip", name=str(region.get("name", "ROI")),
                ))
    elif roi_bounds is not None:
        x0, y0, x1, y1 = roi_bounds
        px0, py0 = x0 / scale, y0 / scale
        px1, py1 = (x1 - 1) / scale, (y1 - 1) / scale
        fig.add_trace(go.Scatter(
            x=[px0, px1, px1, px0, px0], y=[py0, py0, py1, py1, py0],
            mode="lines", line={"color": "#F8FAFC", "width": 2, "dash": "dash"},
            hoverinfo="skip", name="Selected ROI",
        ))

    animation_frames = []
    for i, image in enumerate(video):
        animation_frames.append(go.Frame(
            name=f"frame_{i}",
            data=[go.Heatmap(z=image)],
            traces=[0],
            layout=go.Layout(title={"text": title_for(i)}),
        ))
    fig.frames = animation_frames

    median_interval = float(np.median(np.diff(times))) if len(times) > 1 else 0.25
    frame_ms = max(20, int(1000 * median_interval / max(float(playback_speed), 0.1)))
    slider_steps = [
        {
            "method": "animate",
            "label": f"{int(indices[i]) + 1}",
            "args": [[f"frame_{i}"], {
                "mode": "immediate",
                "frame": {"duration": 0, "redraw": True},
                "transition": {"duration": 0},
            }],
        }
        for i in range(len(video))
    ]
    fig.update_layout(
        template="plotly_dark",
        title={"text": title_for(0), "x": 0.02, "xanchor": "left"},
        height=590,
        margin={"l": 8, "r": 18, "t": 58, "b": 80},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis={"title": "Pixel x", "showgrid": False, "zeroline": False},
        yaxis={"title": "Pixel y", "showgrid": False, "zeroline": False,
               "autorange": "reversed", "scaleanchor": "x", "scaleratio": 1},
        updatemenus=[{
            "type": "buttons", "showactive": False,
            "x": 0.0, "y": -0.12, "xanchor": "left", "yanchor": "top",
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {
                    "fromcurrent": True,
                    "frame": {"duration": frame_ms, "redraw": True},
                    "transition": {"duration": 0},
                }]},
                {"label": "Ⅱ Pause", "method": "animate", "args": [[None], {
                    "mode": "immediate", "frame": {"duration": 0, "redraw": False},
                    "transition": {"duration": 0},
                }]},
            ],
        }],
        sliders=[{
            "active": 0,
            "x": 0.16, "y": -0.12, "len": 0.82,
            "currentvalue": {"prefix": "Frame ", "visible": True},
            "steps": slider_steps,
            "pad": {"t": 30},
        }],
    )
    meta = {
        "original_frames": int(arr.shape[0]),
        "preview_frames": int(len(video)),
        "spatial_stride": int(scale),
        "preview_shape": [int(h), int(w)],
    }
    return fig, meta


def make_arrival_figure(result, max_side: int = 220) -> go.Figure:
    """Compare warming/cooling first-arrival maps with sparse local direction vectors."""
    warm = np.asarray(result.warming_arrival_s, dtype=np.float32)
    cool = np.asarray(result.cooling_arrival_s, dtype=np.float32)
    roi = np.asarray(result.roi_mask, dtype=bool)
    step = max(1, int(math.ceil(max(warm.shape) / max_side)))
    warm = warm[::step, ::step]
    cool = cool[::step, ::step]
    roi_small = roi[::step, ::step]
    warm[~roi_small] = np.nan
    cool[~roi_small] = np.nan

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Warming first-arrival time", "Cooling first-arrival time"),
                        horizontal_spacing=0.08)
    fig.add_trace(go.Heatmap(z=warm, colorscale="Viridis", colorbar={"title": "seconds", "x": 0.45},
                             hovertemplate="x=%{x}<br>y=%{y}<br>arrival=%{z:.2f}s<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=cool, colorscale="Viridis", showscale=False,
                             hovertemplate="x=%{x}<br>y=%{y}<br>arrival=%{z:.2f}s<extra></extra>"), row=1, col=2)

    def vector_segments(vector_tuple, shape):
        vx, vy, valid = vector_tuple
        sy = max(1, shape[0] // 16)
        sx = max(1, shape[1] // 16)
        xs, ys = [], []
        for gy, gx in zip(*np.where(valid)):
            x = gx * sx / step
            y = gy * sy / step
            dx = float(vx[gy, gx]) * 2.8
            dy = float(vy[gy, gx]) * 2.8
            xs.extend([x, x + dx, None])
            ys.extend([y, y + dy, None])
        return xs, ys

    for col, vector in ((1, result.warming_vectors), (2, result.cooling_vectors)):
        xs, ys = vector_segments(vector, roi.shape)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line={"color": "white", "width": 1.4},
                                 hoverinfo="skip", showlegend=False), row=1, col=col)
    fig.update_xaxes(title_text="Pixel x", showgrid=False)
    fig.update_yaxes(title_text="Pixel y", autorange="reversed", showgrid=False, scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(autorange="reversed", showgrid=False, scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_layout(template="plotly_dark", height=480, margin={"l": 8, "r": 8, "t": 55, "b": 16},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def make_metrics_figure(metrics, metric: str = "median", units: str = "°C") -> go.Figure:
    """Create a clean, zoomable time-series plot for ROI metrics."""
    labels = {
        "mean": ("ROI mean", units),
        "median": ("ROI median", units),
        "std": ("ROI standard deviation", units),
        "mean_delta_from_baseline": ("Mean change from baseline", f"Δ {units}"),
        "median_delta_from_baseline": ("Median change from baseline", f"Δ {units}"),
        "mean_event_delta": ("Mean change from event reference", f"Δ {units}"),
        "median_event_delta": ("Median change from event reference", f"Δ {units}"),
        "mean_dT_dt": ("Mean temporal derivative", f"{units}/s"),
        "warming_fraction": ("Warming fraction from baseline", "% of ROI"),
        "cooling_fraction": ("Cooling fraction from baseline", "% of ROI"),
        "event_warming_fraction": ("Warming fraction from event reference", "% of ROI"),
        "event_cooling_fraction": ("Cooling fraction from event reference", "% of ROI"),
    }
    title, ytitle = labels.get(metric, (metric, units))
    y = metrics[metric]
    if metric in {"warming_fraction", "cooling_fraction"}:
        y = y * 100.0
    fig = go.Figure()
    should_fill = metric in {"mean_delta_from_baseline", "median_delta_from_baseline", "warming_fraction", "cooling_fraction"}
    fig.add_trace(go.Scatter(
        x=metrics["time_s"], y=y, mode="lines", name=title,
        line={"color": "#65D9C5", "width": 2.6},
        fill="tozeroy" if should_fill else None,
        fillcolor="rgba(101,217,197,0.10)" if should_fill else None,
        hovertemplate="%{x:.2f} s<br>%{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", height=340,
        title={"text": title, "x": 0.02, "xanchor": "left"},
        xaxis_title="Time (s)", yaxis_title=ytitle,
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified", showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.12)")
    return fig


def make_named_roi_figure(roi_metrics: dict[str, object], metric: str = "median", units: str = "°C",
                          phase_spans: list[dict] | None = None, time_axis_label: str = "Time (s)") -> go.Figure:
    """Plot a selected metric for several named ROIs with protocol phase bands."""
    labels = {
        "mean": ("ROI mean", units), "median": ("ROI median", units),
        "std": ("ROI standard deviation", units),
        "mean_delta_from_baseline": ("Mean change from baseline", f"Δ {units}"),
        "median_delta_from_baseline": ("Median change from baseline", f"Δ {units}"),
        "mean_event_delta": ("Mean change from event reference", f"Δ {units}"),
        "median_event_delta": ("Median change from event reference", f"Δ {units}"),
        "mean_dT_dt": ("Mean temporal derivative", f"{units}/s"),
        "warming_fraction": ("Warming fraction from baseline", "% of ROI"),
        "cooling_fraction": ("Cooling fraction from baseline", "% of ROI"),
        "event_warming_fraction": ("Warming fraction from event reference", "% of ROI"),
        "event_cooling_fraction": ("Cooling fraction from event reference", "% of ROI"),
    }
    title, ytitle = labels.get(metric, (metric, units))
    fig = go.Figure()
    for name, df in roi_metrics.items():
        values = df[metric] * 100.0 if metric in {"warming_fraction", "cooling_fraction", "event_warming_fraction", "event_cooling_fraction"} else df[metric]
        fig.add_trace(go.Scatter(x=df["time_s"], y=values, mode="lines", name=name,
                                 line={"width": 2.3}, hovertemplate=f"{name}<br>%{{x:.2f}} s<br>%{{y:.4f}}<extra></extra>"))
    palette = {"Baseline": "rgba(80,150,220,.09)",
               "Recorded challenge": "rgba(255,176,80,.12)",
               "Recovery": "rgba(80,210,170,.08)",
               "Rewarming · challenge not recorded": "rgba(80,210,170,.08)"}
    for span in phase_spans or []:
        color = palette.get(span["label"], "rgba(150,160,180,.06)")
        fig.add_vrect(x0=span["start_s"], x1=span["end_s"], fillcolor=color, line_width=0,
                      annotation_text=span["label"], annotation_position="top left",
                      annotation_font_size=10)
    fig.update_layout(template="plotly_dark", height=390, title={"text": title, "x": .02},
                      xaxis_title=time_axis_label, yaxis_title=ytitle, hovermode="x unified",
                      margin={"l": 8, "r": 8, "t": 55, "b": 8},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,.12)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,.12)")
    return fig


def make_motion_figure(shifts_yx: np.ndarray) -> go.Figure:
    """Visualize frame-to-frame translation estimates from optional registration."""
    shifts = np.asarray(shifts_yx, dtype=float)
    if shifts.ndim != 2 or shifts.shape[1] != 2:
        raise ValueError("shifts_yx must have shape (frames, 2)")
    fig = go.Figure()
    frames = np.arange(len(shifts))
    fig.add_trace(go.Scatter(x=frames, y=shifts[:, 1], name="X shift", mode="lines", line={"width": 1.8}))
    fig.add_trace(go.Scatter(x=frames, y=shifts[:, 0], name="Y shift", mode="lines", line={"width": 1.8}))
    fig.update_layout(template="plotly_dark", height=280, title="Estimated registration shifts",
                      xaxis_title="Frame", yaxis_title="Shift (px)", hovermode="x unified",
                      margin={"l": 8, "r": 8, "t": 45, "b": 8},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig
