"""Timestamp-aware alignment and descriptive comparison of reference sensor traces.

This module does not convert thermal measurements into perfusion. It aligns
independent recorded signals (e.g. LSCI or laser-Doppler reference traces) to the
thermal frame timebase and reports exploratory association metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import io

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


@dataclass
class ReferenceSignal:
    name: str
    modality: str
    timestamps_s: np.ndarray
    values: np.ndarray
    units: str
    source_name: str = ""
    time_offset_s: float = 0.0
    processing: str = "raw"
    quality: dict = field(default_factory=dict)


def parse_reference_csv(
    payload: bytes,
    time_column: str,
    value_column: str,
    *,
    name: str,
    modality: str,
    units: str,
    source_name: str = "",
    time_offset_s: float = 0.0,
    processing: str = "raw",
    smooth_window_s: float = 1.0,
) -> ReferenceSignal:
    if not payload:
        raise ValueError("Reference signal CSV is empty")
    frame = pd.read_csv(io.BytesIO(payload))
    if time_column not in frame or value_column not in frame:
        raise ValueError("Selected timestamp or signal column is missing from the CSV")
    times = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float)
    input_rows = int(len(frame))
    valid = np.isfinite(times) & np.isfinite(values)
    invalid_rows = int((~valid).sum())
    times, values = times[valid], values[valid]
    non_monotonic_pairs = int(np.sum(np.diff(times) <= 0)) if len(times) > 1 else 0
    if len(times) < 5:
        raise ValueError("At least five finite timestamp/value pairs are required")
    times = times + float(time_offset_s)
    order = np.argsort(times)
    times, values = times[order], values[order]
    # Average duplicate timestamps (e.g. repeated samples in a logger export).
    unique, inverse = np.unique(times, return_inverse=True)
    duplicate_rows = int(len(times) - len(unique))
    if duplicate_rows:
        sums = np.bincount(inverse, weights=values)
        counts = np.bincount(inverse)
        times = unique
        values = sums / counts
    if len(times) < 5 or np.any(np.diff(times) <= 0):
        raise ValueError("Reference timestamps must contain at least five unique, strictly increasing values")
    if processing == "smooth":
        dt = float(np.median(np.diff(times)))
        sigma = max(0.0, float(smooth_window_s) / max(dt, 1e-9) / 2.355)
        if sigma > 0:
            from scipy.ndimage import gaussian_filter1d
            values = gaussian_filter1d(values, sigma=sigma, mode="nearest")
    elif processing != "raw":
        raise ValueError(f"Unsupported signal processing mode: {processing}")
    intervals = np.diff(times)
    median_dt = float(np.median(intervals))
    interval_cv = float(np.std(intervals, ddof=1) / np.mean(intervals)) if len(intervals) > 1 and np.mean(intervals) > 0 else 0.0
    quality = {
        "input_rows": input_rows,
        "valid_rows": int(len(times)),
        "invalid_rows_removed": invalid_rows,
        "duplicate_timestamps_averaged": duplicate_rows,
        "non_monotonic_input_pairs_sorted": non_monotonic_pairs,
        "median_interval_s": median_dt,
        "estimated_sample_rate_hz": float(1.0 / median_dt) if median_dt > 0 else None,
        "interval_coefficient_of_variation": interval_cv,
        "large_interval_gaps": int(np.sum(intervals > 1.5 * median_dt)) if median_dt > 0 else 0,
        "maximum_interval_s": float(np.max(intervals)),
        "duration_s": float(times[-1] - times[0]),
    }
    return ReferenceSignal(name=name, modality=modality, timestamps_s=times, values=values,
                           units=units, source_name=source_name,
                           time_offset_s=float(time_offset_s), processing=processing,
                           quality=quality)


def align_reference_signal(signal: ReferenceSignal, thermal_times_s: np.ndarray) -> np.ndarray:
    thermal_times = np.asarray(thermal_times_s, dtype=float)
    aligned = np.full(thermal_times.shape, np.nan, dtype=float)
    if thermal_times.ndim != 1 or not np.isfinite(thermal_times).all():
        raise ValueError("Thermal timestamps must be a finite one-dimensional array")
    inside = (thermal_times >= signal.timestamps_s[0]) & (thermal_times <= signal.timestamps_s[-1])
    if inside.any():
        aligned[inside] = np.interp(thermal_times[inside], signal.timestamps_s, signal.values)
    return aligned


def compare_reference_signal(
    signal: ReferenceSignal,
    thermal_times_s: np.ndarray,
    thermal_values: np.ndarray,
    max_lag_s: float = 10.0,
    derivative_breaks: tuple[int, ...] | list[int] = (),
) -> tuple[dict, pd.DataFrame]:
    """Compare a reference trace with a thermal ROI feature using real timestamps.

    Lag convention: positive lag means the reference trace is sampled later than
    the thermal feature (reference(t + lag) is paired with thermal(t)). Candidate
    lags are evaluated using timestamp interpolation rather than assuming uniform
    frame intervals. Lag pairing is computed independently inside each continuous
    segment so that no pair crosses a known unrecorded protocol gap.

    Pearson/Spearman values are descriptive; serial dependence is not corrected.
    """
    times = np.asarray(thermal_times_s, dtype=float)
    thermal = np.asarray(thermal_values, dtype=float)
    if thermal.shape != times.shape:
        raise ValueError("Thermal feature and timestamps must have matching shape")
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Thermal timestamps must be finite and strictly increasing")
    if max_lag_s < 0:
        raise ValueError("Maximum lag must be non-negative")
    aligned = align_reference_signal(signal, times)
    valid = np.isfinite(aligned) & np.isfinite(thermal)
    n = int(valid.sum())
    coverage = float(n / max(1, len(times)))
    base = {
        "name": signal.name, "modality": signal.modality, "units": signal.units,
        "source_name": signal.source_name, "time_offset_s": signal.time_offset_s,
        "processing": signal.processing, "quality": signal.quality, "n_aligned": n, "coverage_fraction": coverage,
        "lag_convention": "positive lag: reference signal later than thermal ROI feature",
        "interpretation": "Descriptive / exploratory association only; not evidence that thermal values measure perfusion or that either signal causes the other.",
        "lag_search_respects_protocol_breaks": True,
    }
    stats: dict = dict(base)
    stats.update({"pearson_r": None, "pearson_p_uncorrected": None,
                  "spearman_rho": None, "spearman_p_uncorrected": None,
                  "best_lag_s": None, "best_lag_abs_pearson_r": None})
    if n < 5:
        stats["status"] = "insufficient temporal overlap (<5 aligned samples)"
    elif np.nanstd(aligned[valid]) <= 1e-12 or np.nanstd(thermal[valid]) <= 1e-12:
        stats["status"] = "one signal is constant over the overlap"
    else:
        pr = pearsonr(thermal[valid], aligned[valid])
        sr = spearmanr(thermal[valid], aligned[valid])
        stats.update({"pearson_r": float(pr.statistic), "pearson_p_uncorrected": float(pr.pvalue),
                      "spearman_rho": float(sr.statistic), "spearman_p_uncorrected": float(sr.pvalue),
                      "status": "descriptive statistics computed"})
        dt = float(np.median(np.diff(times)))
        # Segment boundaries are frame indices; boundary frame belongs to the next segment.
        breaks = sorted({int(b) for b in derivative_breaks if 0 < int(b) < len(times)})
        edges = [0, *breaks, len(times)]
        segments = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1) if edges[i + 1] - edges[i] >= 2]
        max_steps = int(min(round(float(max_lag_s) / dt), max(0, len(times) - 5))) if dt > 0 else 0
        candidates = []
        for step in range(-max_steps, max_steps + 1):
            lag = step * dt
            paired_x: list[np.ndarray] = []
            paired_y: list[np.ndarray] = []
            # Interpolate at actual candidate times within the reference signal's support.
            for seg_start, seg_end in segments:
                seg_t = times[seg_start:seg_end]
                seg_x = thermal[seg_start:seg_end]
                query_t = seg_t + lag
                inside = (query_t >= signal.timestamps_s[0]) & (query_t <= signal.timestamps_s[-1])
                if not inside.any():
                    continue
                y = np.full(query_t.shape, np.nan, dtype=float)
                y[inside] = np.interp(query_t[inside], signal.timestamps_s, signal.values)
                keep = np.isfinite(seg_x) & np.isfinite(y)
                if int(keep.sum()) >= 2:
                    paired_x.append(seg_x[keep])
                    paired_y.append(y[keep])
            if not paired_x:
                continue
            x = np.concatenate(paired_x)
            y = np.concatenate(paired_y)
            if len(x) < 5 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            if np.isfinite(r):
                candidates.append((abs(r), lag, r, len(x)))
        if candidates:
            best = max(candidates, key=lambda item: item[0])
            stats.update({"best_lag_s": float(best[1]), "best_lag_abs_pearson_r": float(best[0]),
                          "best_lag_pearson_r_signed": float(best[2]), "best_lag_n": int(best[3]),
                          "lag_segments_used": len(segments)})
    frame = pd.DataFrame({"time_s": times, "thermal_feature": thermal, "reference_signal": aligned})
    return stats, frame


def signal_template_csv() -> bytes:
    frame = pd.DataFrame({"time_s": [0.0, 0.1, 0.2, 0.3, 0.4], "value": [0.0, 0.1, 0.2, 0.1, 0.0]})
    return frame.to_csv(index=False).encode("utf-8")
