"""Exploratory frequency and time-frequency analysis for ROI thermal traces.

The routines require adequately sampled, contiguous signals. They never bridge
known missing protocol intervals and never attach physiological band labels to
frequencies without checking camera sensitivity and acquisition characteristics.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch


def _longest_contiguous_segment(times_s: np.ndarray, values: np.ndarray,
                                derivative_breaks: tuple[int, ...] | list[int] = ()):
    times = np.asarray(times_s, dtype=float)
    vals = np.asarray(values, dtype=float)
    if times.ndim != 1 or vals.shape != times.shape or len(times) < 8:
        raise ValueError("Spectral input needs matching 1-D time/value vectors with at least 8 samples")
    cuts = [0, *sorted({int(x) for x in derivative_breaks if 0 < int(x) < len(times)}), len(times)]
    segments = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        keep = np.isfinite(times[start:end]) & np.isfinite(vals[start:end])
        t, y = times[start:end][keep], vals[start:end][keep]
        if len(t) >= 8:
            segments.append((t, y, start, end))
    if not segments:
        raise ValueError("No contiguous segment has enough finite samples")
    t, y, start, end = max(segments, key=lambda item: len(item[0]))
    if np.any(np.diff(t) <= 0):
        raise ValueError("Spectral timestamps must be strictly increasing")
    dt = np.diff(t)
    median_dt = float(np.median(dt))
    if median_dt <= 0:
        raise ValueError("Invalid sample interval")
    jitter = float(np.std(dt) / median_dt)
    largest_gap_ratio = float(np.max(dt) / median_dt)
    if jitter > 0.10 or largest_gap_ratio > 1.8:
        raise ValueError("Timing is too irregular for the current spectral estimator. Correct timestamps or split dropped-frame segments first.")
    # Interpolate mild jitter to a uniform grid; no extrapolation is performed.
    fs = 1.0 / median_dt
    uniform_t = np.arange(t[0], t[-1] + median_dt * 0.25, median_dt)
    if len(uniform_t) > len(t) * 1.5:
        raise ValueError("Timestamp resampling would materially oversample the data")
    uniform_y = np.interp(uniform_t, t, y)
    meta = {"segment_start_frame": int(start), "segment_end_frame_exclusive": int(end),
            "samples_original": int(len(t)), "samples_uniform": int(len(uniform_t)),
            "sampling_rate_hz": float(fs), "nyquist_hz": float(fs / 2),
            "timestamp_interval_cv": jitter, "max_interval_ratio": largest_gap_ratio,
            "duration_s": float(uniform_t[-1] - uniform_t[0]),
            "warning": "Exploratory thermal-signal spectrum. A spectral peak is not proof of a specific physiological mechanism."}
    return uniform_t, uniform_y, meta


def welch_spectrum(times_s: np.ndarray, values: np.ndarray,
                   derivative_breaks: tuple[int, ...] | list[int] = (),
                   max_frequency_hz: float | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    t, y, meta = _longest_contiguous_segment(times_s, values, derivative_breaks)
    fs = meta["sampling_rate_hz"]
    nyquist = meta["nyquist_hz"]
    nperseg = min(256, len(y))
    frequencies, power = welch(y - np.mean(y), fs=fs, nperseg=nperseg,
                               detrend="linear", scaling="density")
    if max_frequency_hz is not None:
        if max_frequency_hz <= 0:
            raise ValueError("Maximum frequency must be positive")
        keep = frequencies <= min(float(max_frequency_hz), nyquist)
        frequencies, power = frequencies[keep], power[keep]
    meta.update({"method": "Welch PSD", "nperseg": int(nperseg),
                 "frequency_limit_hz": float(frequencies[-1]) if len(frequencies) else None})
    if len(y) / fs < 5:
        meta["warning"] += " Recording segment is under five seconds; low-frequency estimates have poor resolution."
    return frequencies, power, meta


def wavelet_scalogram(times_s: np.ndarray, values: np.ndarray,
                      derivative_breaks: tuple[int, ...] | list[int] = (),
                      max_frequency_hz: float | None = None,
                      wavelet: str = "cmor1.5-1.0",
                      n_frequencies: int = 48):
    try:
        import pywt
    except ImportError as exc:
        raise RuntimeError("Wavelet scalograms are optional. Install with: pip install -e '.[wavelets]'") from exc
    t, y, meta = _longest_contiguous_segment(times_s, values, derivative_breaks)
    fs = meta["sampling_rate_hz"]
    nyquist = meta["nyquist_hz"]
    duration = meta["duration_s"]
    f_hi = min(float(max_frequency_hz), nyquist) if max_frequency_hz else nyquist
    f_lo = max(1.0 / max(duration, 1e-9), 0.005)
    if f_hi <= f_lo:
        raise ValueError(f"Insufficient duration/sample rate for wavelet analysis: usable band {f_lo:.4g}–{f_hi:.4g} Hz")
    frequencies = np.geomspace(f_lo, f_hi, int(n_frequencies))
    central = pywt.central_frequency(wavelet)
    scales = central * fs / frequencies
    coefficients, actual_frequencies = pywt.cwt(y - np.mean(y), scales, wavelet,
                                                  sampling_period=1.0/fs)
    power = np.abs(coefficients) ** 2
    meta.update({"method": "Continuous wavelet transform", "wavelet": wavelet,
                 "frequency_min_hz": float(np.min(actual_frequencies)),
                 "frequency_max_hz": float(np.max(actual_frequencies)),
                 "frequency_bins": int(len(actual_frequencies))})
    if duration * float(np.min(actual_frequencies)) < 3:
        meta["warning"] += " Fewer than three cycles are present at the lowest displayed frequency; interpret low-frequency edge regions cautiously."
    return t, np.asarray(actual_frequencies), power, meta
