"""Optional exponential recovery fit for controlled cooling/rewarming protocols."""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def fit_rewarming_time_constant(times_s: np.ndarray, roi_temperature: np.ndarray, start_index: int = 0) -> dict:
    """Fit T(t)=T_inf - A exp(-(t-t0)/tau) after the observed temperature minimum.

    Only a descriptive protocol metric. The fit is not a direct perfusion measurement.
    At least 8 post-minimum samples and measurable recovery are required.
    """
    t = np.asarray(times_s, dtype=float)
    y = np.asarray(roi_temperature, dtype=float)
    if t.ndim != 1 or y.shape != t.shape or len(t) < 10 or np.any(np.diff(t) <= 0):
        return {"valid": False, "reason": "Need at least 10 aligned samples and strictly increasing times."}
    start_index = int(start_index)
    if not (0 <= start_index < len(t) - 7):
        return {"valid": False, "reason": "start_index must leave at least 8 samples in the recovery segment."}
    t_segment, y_segment = t[start_index:], y[start_index:]
    if not np.isfinite(y_segment).all():
        return {"valid": False, "reason": "Recovery segment contains non-finite ROI values."}
    idx0 = int(np.argmin(y_segment))
    tfit, yfit = t_segment[idx0:], y_segment[idx0:]
    if len(tfit) < 8:
        return {"valid": False, "reason": "Fewer than 8 samples remain after the temperature minimum."}
    span = float(np.nanmax(yfit) - np.nanmin(yfit))
    if not np.isfinite(span) or span < 1e-6:
        return {"valid": False, "reason": "Insufficient rewarming amplitude after the minimum."}
    t0 = float(tfit[0])

    def model(x, t_inf, amplitude, tau):
        return t_inf - amplitude * np.exp(-np.maximum(x - t0, 0.0) / tau)

    try:
        p0 = [float(yfit[-1]), max(span, 1e-3), max((tfit[-1] - tfit[0]) / 3, 1e-3)]
        params, _ = curve_fit(model, tfit, yfit, p0=p0,
                              bounds=([np.nanmin(yfit) - span, 0.0, 1e-4],
                                      [np.nanmax(yfit) + 2 * span, 4 * span, max(tfit[-1] - tfit[0], 1e-3) * 10]),
                              maxfev=20000)
        pred = model(tfit, *params)
        ss_res = float(np.sum((yfit - pred) ** 2))
        ss_tot = float(np.sum((yfit - np.mean(yfit)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return {"valid": True, "recovery_start_frame": start_index, "t0_s": t0,
                "asymptote": float(params[0]), "amplitude": float(params[1]),
                "tau_s": float(params[2]), "r_squared": r2,
                "note": "Descriptive thermal recovery time constant; not a calibrated blood-perfusion value."}
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        return {"valid": False, "reason": f"Exponential fit did not converge: {exc}"}
