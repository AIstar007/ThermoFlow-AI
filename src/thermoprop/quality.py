"""Acquisition and processing quality-control summaries (not clinical quality grades)."""
from __future__ import annotations

import numpy as np


def compute_quality_report(
    frames: np.ndarray,
    times_s: np.ndarray,
    baseline_frames: int,
    roi_mask: np.ndarray,
    registration_shifts_yx_px: np.ndarray | None = None,
    roi_overlap_fraction: float = 0.0,
) -> dict:
    arr = np.asarray(frames, dtype=np.float32)
    times = np.asarray(times_s, dtype=float)
    roi = np.asarray(roi_mask, dtype=bool)
    n, h, w = arr.shape
    dt = np.diff(times)
    dt_med = float(np.median(dt)) if dt.size else float("nan")
    dt_cv = float(np.std(dt) / dt_med) if dt_med > 0 else float("nan")
    finite_count = 0
    for frame in arr:
        finite_count += int(np.isfinite(frame).sum())
    finite_fraction = float(finite_count / arr.size)
    roi_fraction = float(roi.mean())
    baseline_drift = float("nan")
    baseline_medians = []
    for idx in range(min(int(baseline_frames), n)):
        values = arr[idx][roi]
        values = values[np.isfinite(values)]
        baseline_medians.append(float(np.median(values)) if values.size else float("nan"))
    if len(baseline_medians) >= 3 and np.isfinite(baseline_medians).all():
        slope = np.polyfit(times[:len(baseline_medians)], baseline_medians, 1)[0]
        baseline_drift = float(slope * 60.0)  # per minute in original input units

    shifts = np.zeros((n, 2), dtype=float) if registration_shifts_yx_px is None else np.asarray(registration_shifts_yx_px, dtype=float)
    if shifts.shape != (n, 2):
        shifts = np.zeros((n, 2), dtype=float)
    magnitudes = np.linalg.norm(shifts, axis=1)
    # Estimate global outlier fences from a bounded, spatial-temporal sample so
    # QC does not allocate a second full-size copy of a long high-resolution video.
    sample = arr[::max(1, n // 32), ::max(1, h // 128), ::max(1, w // 128)]
    finite_values = sample[np.isfinite(sample)]
    if finite_values.size:
        q1, q3 = np.nanpercentile(finite_values, [25, 75])
        iqr = float(q3 - q1)
        if iqr > 1e-9:
            outlier_fraction = float(np.mean((finite_values < q1 - 3.0 * iqr) | (finite_values > q3 + 3.0 * iqr)))
        else:
            outlier_fraction = 0.0
        sequence_iqr = iqr
    else:
        outlier_fraction = float("nan")
        sequence_iqr = float("nan")
    drift_fraction_of_iqr = (abs(baseline_drift) / sequence_iqr
                              if np.isfinite(baseline_drift) and np.isfinite(sequence_iqr) and sequence_iqr > 1e-9
                              else float("nan"))
    warnings = []
    if finite_fraction < 0.999:
        warnings.append("Input contains non-finite samples; verify sensor invalid-pixel flags and preprocessing fill behaviour.")
    if roi_fraction < 0.02:
        warnings.append("The union of selected ROIs covers less than 2% of the image; regional statistics may be unstable.")
    if dt_cv > 0.05:
        warnings.append("Frame-interval variability exceeds 5%; verify timestamp alignment and dropped frames.")
    if np.isfinite(drift_fraction_of_iqr) and drift_fraction_of_iqr > 0.02:
        warnings.append("Baseline median drift exceeds 2% of the sequence interquartile range per minute; acclimatization, ambient drift, motion, or sensor drift may influence the sequence.")
    if np.isfinite(outlier_fraction) and outlier_fraction > 0.005:
        warnings.append("More than 0.5% of samples lie beyond the global 3×IQR fences; inspect for transient artefacts or invalid pixels.")
    if magnitudes.size and float(np.max(magnitudes)) > 0.05 * min(h, w):
        warnings.append("Estimated maximum translation exceeds 5% of the image side; inspect the alignment overlays and avoid interpreting registration as anatomical tracking.")
    if roi_overlap_fraction > 0:
        warnings.append("Named ROIs overlap. Individual ROI results remain valid, but their areas should not be summed as independent tissue regions.")
    return {
        "frames": int(n), "height": int(h), "width": int(w),
        "finite_sample_fraction": finite_fraction,
        "roi_coverage_fraction": roi_fraction,
        "roi_coverage_pixels": int(roi.sum()),
        "median_frame_interval_s": dt_med,
        "frame_interval_cv": dt_cv,
        "max_frame_interval_s": float(np.max(dt)) if dt.size else None,
        "baseline_median_drift_per_min_input_units": baseline_drift,
        "baseline_drift_fraction_of_sequence_iqr_per_min": drift_fraction_of_iqr,
        "median_registration_shift_px": float(np.median(magnitudes)),
        "max_registration_shift_px": float(np.max(magnitudes)),
        "sampled_outlier_pixel_fraction_3_iqr": outlier_fraction,
        "sampled_sequence_iqr_input_units": sequence_iqr,
        "outlier_qc_sample_shape": [int(v) for v in sample.shape],
        "roi_overlap_fraction_of_union": float(roi_overlap_fraction),
        "warnings": warnings,
        "disclaimer": "Engineering QC indicators only; not a clinical quality grade or perfusion validation.",
    }
