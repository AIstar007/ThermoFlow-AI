"""Publication-style summary visualizations."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import AnalysisResult


def create_summary_figure(result: AnalysisResult, frames: np.ndarray, output_path: str | Path, unit_label: str = "°C") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    ax = axes.ravel()
    first_dynamic_idx = min(result.summary["baseline_frames"], len(frames) - 1)
    roi = result.roi_mask

    im0 = ax[0].imshow(frames[first_dynamic_idx], cmap="inferno")
    ax[0].contour(roi, levels=[0.5], colors="white", linewidths=0.8)
    ax[0].set_title("First post-baseline thermal frame")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, label=unit_label)

    last_delta = np.where(roi, result.delta[-1], np.nan)
    lim = float(np.nanpercentile(np.abs(result.delta[:, roi]), 98))
    lim = max(lim, 1e-6)
    im1 = ax[1].imshow(last_delta, cmap="coolwarm", vmin=-lim, vmax=lim)
    ax[1].set_title("Final Δ from baseline")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, label=f"Δ {unit_label}")

    im2 = ax[2].imshow(np.where(roi, result.warming_arrival_s, np.nan), cmap="viridis")
    vx, vy, wvalid = result.warming_vectors
    yy, xx = np.mgrid[0:vx.shape[0], 0:vx.shape[1]]
    # Vector arrays are sparse-grid coordinates; x is right and y is down, as in imshow.
    step_y = max(1, roi.shape[0] // 16)
    step_x = max(1, roi.shape[1] // 16)
    ax[2].quiver(xx * step_x, yy * step_y, np.where(wvalid, vx, np.nan), np.where(wvalid, vy, np.nan),
                 color="white", angles="xy", scale_units="xy", scale=0.35, width=0.004)
    ax[2].set_title("Warming first-arrival map + direction")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, label="s")

    im3 = ax[3].imshow(np.where(roi, result.cooling_arrival_s, np.nan), cmap="viridis")
    cvx, cvy, cvalid = result.cooling_vectors
    yy2, xx2 = np.mgrid[0:cvx.shape[0], 0:cvx.shape[1]]
    ax[3].quiver(xx2 * step_x, yy2 * step_y, np.where(cvalid, cvx, np.nan), np.where(cvalid, cvy, np.nan),
                 color="white", angles="xy", scale_units="xy", scale=0.35, width=0.004)
    ax[3].set_title("Cooling first-arrival map + direction")
    fig.colorbar(im3, ax=ax[3], fraction=0.046, label="s")

    m = result.metrics
    ax[4].plot(m.time_s, m["mean"], label="ROI mean")
    ax[4].plot(m.time_s, m["median"], label="ROI median", alpha=0.8)
    ax[4].axvline(m.time_s.iloc[result.summary["baseline_frames"]], linestyle="--", color="grey", label="baseline end")
    ax[4].set_title("ROI temperature/intensity over time")
    ax[4].set_xlabel("Time (s)")
    ax[4].set_ylabel(unit_label)
    ax[4].legend()
    ax[4].grid(alpha=0.2)

    ax[5].plot(m.time_s, m["mean_delta_from_baseline"], label="Mean Δ")
    ax[5].set_title("ROI change and rate")
    ax[5].set_xlabel("Time (s)")
    ax[5].set_ylabel(f"Δ {unit_label}")
    rate_ax = ax[5].twinx()
    rate_ax.plot(m.time_s, m["mean_dT_dt"], label="Mean dT/dt", alpha=0.8)
    rate_ax.set_ylabel(f"{unit_label}/s")
    lines = ax[5].get_lines() + rate_ax.get_lines()
    ax[5].legend(lines, [line.get_label() for line in lines], loc="best")
    ax[5].grid(alpha=0.2)

    for a in (ax[0], ax[1], ax[2], ax[3]):
        a.set_axis_off()
    fig.suptitle("Thermal Pattern Propagation Analysis\nApparent surface-temperature dynamics — not direct blood-flow measurement", fontsize=14)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_json(data: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(_clean_json(data), indent=2, allow_nan=False, default=_json_default), encoding="utf-8")


def _clean_json(value):
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)}")
