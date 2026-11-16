"""Participant/session/ROI-level features for grouped reference-model training."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid


def build_participant_feature_table(regional_metrics: dict[str, pd.DataFrame],
                                    regional_fits: dict[str, dict],
                                    phase_labels: np.ndarray,
                                    acquisition_metadata: dict | None = None,
                                    participant_id: str = "FILL_IN",
                                    session_id: str = "FILL_IN") -> pd.DataFrame:
    """Create one row per named ROI (never one row per video frame).

    The reference label is intentionally blank; the user must add the paired,
    independently measured sensor result for the matching ROI/session.
    """
    phases = np.asarray(phase_labels, dtype=object)
    meta = acquisition_metadata or {}
    rows = []
    for roi_name, frame in regional_metrics.items():
        baseline = frame.loc[phases == "Baseline"] if len(phases) == len(frame) else frame.iloc[:max(2, len(frame)//5)]
        recovery_mask = np.array([str(p).startswith("Recovery") or str(p).startswith("Rewarming") for p in phases])
        recovery = frame.loc[recovery_mask] if len(recovery_mask) == len(frame) and recovery_mask.any() else frame.iloc[max(2, len(frame)//5):]
        time = recovery["time_s"].to_numpy(dtype=float) if len(recovery) else np.array([])
        event_delta = recovery["median_event_delta"].to_numpy(dtype=float) if "median_event_delta" in recovery else np.array([])
        rates = recovery["mean_dT_dt"].to_numpy(dtype=float) if "mean_dT_dt" in recovery else np.array([])
        finite_delta = np.isfinite(event_delta) if event_delta.size else np.array([], dtype=bool)
        auc = float(trapezoid(event_delta[finite_delta], time[finite_delta])) if finite_delta.sum() >= 2 else np.nan
        finite_rates = rates[np.isfinite(rates)] if rates.size else np.array([])
        positive_rates = finite_rates[finite_rates > 0]
        negative_rates = finite_rates[finite_rates < 0]
        warm = recovery["event_warming_fraction"].to_numpy(dtype=float) if "event_warming_fraction" in recovery else np.array([])
        cool = recovery["event_cooling_fraction"].to_numpy(dtype=float) if "event_cooling_fraction" in recovery else np.array([])
        fit = regional_fits.get(roi_name, {})
        baseline_median = baseline["median"].to_numpy(dtype=float) if len(baseline) and "median" in baseline else np.array([])
        baseline_std = baseline["std"].to_numpy(dtype=float) if len(baseline) and "std" in baseline else np.array([])
        rows.append({
            "participant_id": participant_id,
            "session_id": session_id,
            "roi_name": roi_name,
            "reference_perfusion": np.nan,
            "mean_temperature": float(np.nanmedian(baseline_median)) if np.isfinite(baseline_median).any() else np.nan,
            "temperature_sd": float(np.nanmedian(baseline_std)) if np.isfinite(baseline_std).any() else np.nan,
            "mean_event_delta": float(np.nanmean(event_delta)) if finite_delta.any() else np.nan,
            "max_warming_rate": float(np.nanmax(positive_rates)) if positive_rates.size else np.nan,
            "max_cooling_rate": float(abs(np.nanmin(negative_rates))) if negative_rates.size else np.nan,
            "rewarming_tau_s": float(fit.get("tau_s")) if fit.get("valid") and fit.get("tau_s") is not None else np.nan,
            "event_temperature_auc": auc,
            "warming_coverage": float(np.nanmean(warm)) if np.isfinite(warm).any() else np.nan,
            "cooling_coverage": float(np.nanmean(cool)) if np.isfinite(cool).any() else np.nan,
            "ambient_temperature_c": meta.get("ambient_temperature_c", np.nan),
            "relative_humidity_percent": meta.get("relative_humidity_percent", np.nan),
            "acclimatization_min": meta.get("acclimatization_minutes", np.nan),
            "pixel_value_units": meta.get("pixel_value_units", "unknown"),
            "radiometric_calibration_confirmed": bool(meta.get("radiometric_calibration_confirmed", False)),
        })
    return pd.DataFrame(rows)
