"""Interpretable analysis of temperature change and apparent thermal-front propagation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d


@dataclass
class AnalysisResult:
    baseline: np.ndarray
    delta: np.ndarray
    event_delta: np.ndarray
    delta_rate: np.ndarray
    roi_mask: np.ndarray
    metrics: pd.DataFrame
    warming_arrival_s: np.ndarray
    cooling_arrival_s: np.ndarray
    warming_vectors: tuple[np.ndarray, np.ndarray, np.ndarray]
    cooling_vectors: tuple[np.ndarray, np.ndarray, np.ndarray]
    summary: dict


def rectangular_roi(shape: tuple[int, int], x0: int = 0, y0: int = 0, x1: int | None = None, y1: int | None = None) -> np.ndarray:
    h, w = shape
    x1 = w if x1 is None else int(x1)
    y1 = h if y1 is None else int(y1)
    x0, y0 = int(x0), int(y0)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(f"ROI must satisfy 0 <= x0 < x1 <= {w}, 0 <= y0 < y1 <= {h}; and 0 <= y0 < y1 <= {h}")
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _robust_noise_sigma(baseline_frames: np.ndarray, baseline: np.ndarray) -> float:
    residual = baseline_frames - baseline[None, :, :]
    values = residual[np.isfinite(residual)]
    if values.size == 0:
        return 0.0
    med = np.median(values)
    return float(1.4826 * np.median(np.abs(values - med)))


def _arrival_map(delta: np.ndarray, times: np.ndarray, mask: np.ndarray, threshold: float, sign: int) -> np.ndarray:
    """First time each pixel crosses +threshold (sign=+1) or -threshold (sign=-1)."""
    crossed = (sign * delta >= threshold) & mask[None, :, :]
    occurred = crossed.any(axis=0)
    first_idx = np.argmax(crossed, axis=0)
    arrival = np.full(mask.shape, np.nan, dtype=np.float32)
    if occurred.any():
        arrival[occurred] = times[first_idx[occurred]].astype(np.float32)
    return arrival


def _propagation_vectors(
    arrival: np.ndarray,
    roi_mask: np.ndarray,
    pixel_size_mm: float | None,
    smooth_sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return local propagation vectors and robust speed/direction summaries.

    The gradient points toward pixels reached later by the front; its inverse norm
    estimates the normal speed of an apparent thermal front, not blood velocity.
    """
    valid = np.isfinite(arrival) & roi_mask
    nan_map = np.where(valid, arrival, 0.0).astype(np.float32)
    weights = valid.astype(np.float32)
    den = gaussian_filter(weights, sigma=smooth_sigma, mode="constant")
    num = gaussian_filter(nan_map, sigma=smooth_sigma, mode="constant")
    smooth = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0.55)
    # Require central-difference support on both sides; edge artefacts are masked.
    filled = np.where(np.isfinite(smooth), smooth, 0.0)
    if pixel_size_mm is None:
        sy = sx = 1.0
        unit = "px/s"
    else:
        sy = sx = float(pixel_size_mm)
        unit = "mm/s"
    grad_y, grad_x = np.gradient(filled, sy, sx)
    magnitude = np.hypot(grad_x, grad_y)
    grad_valid = valid & np.isfinite(smooth) & (den > 0.72) & (magnitude > 1e-5)
    vx = np.where(grad_valid, grad_x / np.maximum(magnitude, 1e-12), np.nan)
    vy = np.where(grad_valid, grad_y / np.maximum(magnitude, 1e-12), np.nan)
    speed = np.where(grad_valid, 1.0 / np.maximum(magnitude, 1e-12), np.nan)
    if grad_valid.any():
        ux = float(np.nanmean(vx))
        uy = float(np.nanmean(vy))
        norm = float(np.hypot(ux, uy))
        coherence = min(1.0, norm)
        if norm > 0:
            ux, uy = ux / norm, uy / norm
        median_speed = float(np.nanmedian(speed[grad_valid]))
        q25, q75 = [float(v) for v in np.nanpercentile(speed[grad_valid], [25, 75])]
        angle = float(np.degrees(np.arctan2(uy, ux)))
        direction = _direction_label(ux, uy)
    else:
        ux = uy = angle = median_speed = q25 = q75 = coherence = float("nan")
        direction = "insufficient detected front"
    summary = {
        "direction": direction,
        "direction_angle_deg_image_coordinates": angle,
        "mean_direction_unit_x": ux,
        "mean_direction_unit_y": uy,
        "median_front_speed": median_speed,
        "front_speed_q25": q25,
        "front_speed_q75": q75,
        "speed_unit": unit,
        "direction_coherence_0_to_1": coherence,
        "arrival_pixels": int(valid.sum()),
    }
    # Reduce displayed vectors to a sparse grid while preserving NaNs.
    step_y = max(1, arrival.shape[0] // 16)
    step_x = max(1, arrival.shape[1] // 16)
    return vx[::step_y, ::step_x], vy[::step_y, ::step_x], grad_valid[::step_y, ::step_x], summary


def _direction_label(ux: float, uy: float) -> str:
    # Image coordinates: +x right, +y down.
    if abs(ux) >= 2 * abs(uy):
        return "right" if ux >= 0 else "left"
    if abs(uy) >= 2 * abs(ux):
        return "down" if uy >= 0 else "up"
    if ux >= 0 and uy < 0:
        return "up-right"
    if ux < 0 and uy < 0:
        return "up-left"
    if ux >= 0 and uy >= 0:
        return "down-right"
    return "down-left"


def analyze_sequence(
    frames: np.ndarray,
    fps: float = 1.0,
    times_s: np.ndarray | None = None,
    roi_mask: np.ndarray | None = None,
    baseline_frames: int = 5,
    change_threshold: float = 0.25,
    min_threshold: float = 0.15,
    pixel_size_mm: float | None = None,
    temporal_smoothing_sigma: float = 0.0,
    derivative_breaks: tuple[int, ...] | list[int] = (),
    front_reference_frame: int | None = None,
    front_reference_frames: int = 5,
) -> AnalysisResult:
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] < 3:
        raise ValueError("Expected at least three frames with shape (frames, height, width)")
    n, h, w = arr.shape
    if times_s is None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        times = np.arange(n, dtype=float) / float(fps)
    else:
        times = np.asarray(times_s, dtype=float)
        if times.shape != (n,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("times_s must contain one finite, strictly increasing timestamp per frame")
    baseline_frames = int(baseline_frames)
    if not (2 <= baseline_frames < n):
        raise ValueError("baseline_frames must be at least 2 and less than the number of frames")
    if change_threshold < 0 or min_threshold < 0:
        raise ValueError("Thresholds must be non-negative")
    if pixel_size_mm is not None and pixel_size_mm <= 0:
        raise ValueError("pixel_size_mm must be > 0")

    if roi_mask is None:
        roi = np.ones((h, w), dtype=bool)
    else:
        roi = np.asarray(roi_mask, dtype=bool)
        if roi.shape != (h, w) or not roi.any():
            raise ValueError("roi_mask must be a non-empty mask matching each frame")

    baseline = np.nanmedian(arr[:baseline_frames], axis=0).astype(np.float32)
    delta = (arr - baseline[None, :, :]).astype(np.float32)
    # A separate event reference is needed for recovery recordings where the
    # cooling challenge occurred before the recorded rewarming sequence. In that
    # case, pixels may warm substantially while remaining below the pre-challenge
    # baseline; a baseline-relative positive threshold would miss that recovery.
    if front_reference_frame is None:
        ref_start = 0
        event_reference = baseline
    else:
        ref_start = int(front_reference_frame)
        if not (0 <= ref_start < n):
            raise ValueError("front_reference_frame must be within the sequence")
        ref_count = int(front_reference_frames)
        if ref_count < 1 or ref_start + ref_count > n:
            raise ValueError("front_reference_frames must select at least one available frame")
        event_reference = np.nanmedian(arr[ref_start:ref_start + ref_count], axis=0).astype(np.float32)
    event_delta = (arr - event_reference[None, :, :]).astype(np.float32)

    # Do not smooth or differentiate across acquisition discontinuities (for
    # example, the unrecorded cooling challenge between baseline and rewarming).
    breaks = sorted({int(v) for v in derivative_breaks if 0 < int(v) < n})
    edges = [0, *breaks, n]
    if temporal_smoothing_sigma > 0:
        for seg_start, seg_end in zip(edges[:-1], edges[1:]):
            delta[seg_start:seg_end] = gaussian_filter1d(
                delta[seg_start:seg_end], sigma=temporal_smoothing_sigma, axis=0, mode="nearest"
            )
        event_edges = sorted(set(edges + ([ref_start] if 0 < ref_start < n else [])))
        for seg_start, seg_end in zip(event_edges[:-1], event_edges[1:]):
            event_delta[seg_start:seg_end] = gaussian_filter1d(
                event_delta[seg_start:seg_end], sigma=temporal_smoothing_sigma, axis=0, mode="nearest"
            )
    # Pre-event frames are not valid samples of the event-relative channel. Mask
    # them so a warm pre-challenge baseline cannot be mistaken for early recovery.
    if ref_start > 0:
        event_delta[:ref_start] = np.nan
    delta_rate = np.full_like(delta, np.nan, dtype=np.float32)
    for seg_start, seg_end in zip(edges[:-1], edges[1:]):
        if seg_end - seg_start >= 2:
            delta_rate[seg_start:seg_end] = np.gradient(
                delta[seg_start:seg_end], times[seg_start:seg_end], axis=0
            ).astype(np.float32)
    noise_sigma = _robust_noise_sigma(arr[:baseline_frames], baseline)
    adaptive_threshold = max(float(change_threshold), float(min_threshold), 3.0 * noise_sigma)

    # Per-frame statistics are calculated only within the selected ROI.
    rows = []
    roi_pixels = int(roi.sum())
    for idx in range(n):
        values = arr[idx][roi]
        changes = delta[idx][roi]
        event_changes = event_delta[idx][roi]
        rates = delta_rate[idx][roi]
        rows.append({
            "frame": idx,
            "time_s": float(times[idx]),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
            "mean_delta_from_baseline": float(np.mean(changes)),
            "median_delta_from_baseline": float(np.median(changes)),
            "mean_event_delta": float(np.nanmean(event_changes)) if np.isfinite(event_changes).any() else float("nan"),
            "median_event_delta": float(np.nanmedian(event_changes)) if np.isfinite(event_changes).any() else float("nan"),
            "mean_dT_dt": float(np.nanmean(rates)) if np.isfinite(rates).any() else float("nan"),
            "warming_fraction": float(np.mean(changes >= adaptive_threshold)),
            "cooling_fraction": float(np.mean(changes <= -adaptive_threshold)),
            "event_warming_fraction": float(np.mean(event_changes >= adaptive_threshold)) if np.isfinite(event_changes).any() else float("nan"),
            "event_cooling_fraction": float(np.mean(event_changes <= -adaptive_threshold)) if np.isfinite(event_changes).any() else float("nan"),
            "roi_pixels": roi_pixels,
        })
    metrics = pd.DataFrame(rows)
    # Propagation is evaluated relative to the selected event reference, while
    # baseline-relative Δ remains available as a separate analysis channel.
    warming_arrival = _arrival_map(event_delta, times, roi, adaptive_threshold, +1)
    cooling_arrival = _arrival_map(event_delta, times, roi, adaptive_threshold, -1)
    warm_vectors = _propagation_vectors(warming_arrival, roi, pixel_size_mm)
    cool_vectors = _propagation_vectors(cooling_arrival, roi, pixel_size_mm)

    roi_changes = delta[:, roi]
    event_fraction_warm = float(np.isfinite(warming_arrival).sum() / roi_pixels)
    event_fraction_cool = float(np.isfinite(cooling_arrival).sum() / roi_pixels)
    summary = {
        "n_frames": int(n), "height": int(h), "width": int(w),
        "duration_s": float(times[-1] - times[0]), "fps_effective": float(1.0 / np.median(np.diff(times))),
        "baseline_frames": baseline_frames, "derivative_breaks": breaks, "baseline_mean": float(np.mean(baseline[roi])),
        "front_reference_frame": int(ref_start), "front_reference_frames": int(front_reference_frames if front_reference_frame is not None else baseline_frames),
        "front_reference_time_s": float(times[ref_start]),
        "front_reference_description": "event-relative reference for front-arrival maps; separate from the pre-protocol baseline",
        "baseline_median": float(np.median(baseline[roi])), "baseline_noise_sigma_robust": noise_sigma,
        "applied_change_threshold": adaptive_threshold,
        "max_roi_median_temperature_change": float(np.max(np.abs(metrics["median_delta_from_baseline"].to_numpy()))),
        "warming_event_fraction": event_fraction_warm,
        "cooling_event_fraction": event_fraction_cool,
        "event_warming_fraction_mean": float(metrics["event_warming_fraction"].mean()),
        "event_cooling_fraction_mean": float(metrics["event_cooling_fraction"].mean()),
        "overall_classification": _overall_change_label(metrics, adaptive_threshold),
        "warming_propagation": warm_vectors[3],
        "cooling_propagation": cool_vectors[3],
        "interpretation": "Apparent thermal-front propagation only; not a measurement of blood-flow direction or perfusion.",
    }
    return AnalysisResult(baseline, delta, event_delta, delta_rate, roi, metrics, warming_arrival, cooling_arrival,
                          (warm_vectors[0], warm_vectors[1], warm_vectors[2]),
                          (cool_vectors[0], cool_vectors[1], cool_vectors[2]), summary)


def _overall_change_label(metrics: pd.DataFrame, threshold: float) -> str:
    tail = metrics.iloc[-max(2, len(metrics) // 5):]
    warm = float(tail["warming_fraction"].mean())
    cool = float(tail["cooling_fraction"].mean())
    if warm > max(0.03, cool * 1.5):
        return "predominantly warming"
    if cool > max(0.03, warm * 1.5):
        return "predominantly cooling"
    return "mixed or no thresholded change"
