"""Optional Lab Streaming Layer (LSL) discovery and finite sensor capture.

LSL provides timestamped stream exchange and clock-offset measurements. It does
not magically synchronize a camera that writes timestamps on an unrelated clock;
when a camera is not itself recorded through LSL, document and apply a measured
clock offset before comparing signals.
"""
from __future__ import annotations

import csv
import io
import time
from typing import Any

import numpy as np


def _pylsl():
    try:
        import pylsl
    except ImportError as exc:
        raise RuntimeError("LSL support is optional. Install it with: pip install -e '.[sensors]'") from exc
    return pylsl


def _safe_call(obj: Any, method: str, default=None):
    try:
        value = getattr(obj, method)()
        return value if value is not None else default
    except Exception:
        return default


def list_lsl_streams(wait_time_s: float = 1.0) -> list[dict[str, Any]]:
    """Discover visible LSL outlets without opening or consuming their data."""
    if not 0.1 <= float(wait_time_s) <= 10.0:
        raise ValueError("LSL discovery wait must be between 0.1 and 10 seconds.")
    pylsl = _pylsl()
    infos = pylsl.resolve_streams(wait_time=float(wait_time_s))
    output = []
    for info in infos:
        output.append({
            "name": _safe_call(info, "name", "unnamed"),
            "type": _safe_call(info, "type", ""),
            "source_id": _safe_call(info, "source_id", ""),
            "uid": _safe_call(info, "uid", ""),
            "hostname": _safe_call(info, "hostname", ""),
            "channel_count": int(_safe_call(info, "channel_count", 1) or 1),
            "nominal_srate": float(_safe_call(info, "nominal_srate", 0.0) or 0.0),
            "channel_format": _safe_call(info, "channel_format", ""),
        })
    return output


def capture_lsl_channel(
    source_id: str | None = None,
    *,
    stream_uid: str | None = None,
    channel_index: int = 0,
    duration_s: float = 10.0,
    max_samples: int = 2_000_000,
    inlet_timeout_s: float = 5.0,
    clock_correct: bool = True,
) -> dict[str, Any]:
    """Capture one numeric channel from a discovered LSL stream for finite time.

    Timestamps are corrected into the receiving computer's LSL clock domain
    when requested. Keep the returned clock-correction metadata with the data.
    """
    source_id = str(source_id or "").strip()
    stream_uid = str(stream_uid or "").strip()
    if not source_id and not stream_uid:
        raise ValueError("Provide an LSL stream_uid or source_id.")
    if not 0.1 <= float(duration_s) <= 3600:
        raise ValueError("LSL capture duration must be between 0.1 and 3600 seconds.")
    if int(max_samples) < 5 or int(max_samples) > 20_000_000:
        raise ValueError("max_samples must be between 5 and 20,000,000.")
    pylsl = _pylsl()
    lookup_property = "uid" if stream_uid else "source_id"
    lookup_value = stream_uid or source_id
    infos = pylsl.resolve_byprop(lookup_property, lookup_value, minimum=1, timeout=float(inlet_timeout_s))
    if not infos:
        raise RuntimeError(f"No LSL stream found for {lookup_property}={lookup_value!r}.")
    info = infos[0]
    channels = int(_safe_call(info, "channel_count", 1) or 1)
    if not 0 <= int(channel_index) < channels:
        raise ValueError(f"Channel index must be between 0 and {channels - 1}.")
    inlet = pylsl.StreamInlet(info, max_buflen=max(60, int(float(duration_s) + 10)))
    samples_out: list[np.ndarray] = []
    timestamps_out: list[np.ndarray] = []
    correction = 0.0
    correction_history: list[dict[str, float]] = []
    total_samples = 0
    started = time.monotonic()
    next_correction_at = 0.0
    try:
        inlet.open_stream(timeout=float(inlet_timeout_s))
        if clock_correct:
            try:
                candidate = float(inlet.time_correction(timeout=float(inlet_timeout_s)))
                if np.isfinite(candidate):
                    correction = candidate
                    correction_history.append({"elapsed_capture_s": 0.0, "correction_s": correction})
            except Exception:
                correction = 0.0
            next_correction_at = time.monotonic() + 5.0
        while time.monotonic() - started < float(duration_s):
            if clock_correct and time.monotonic() >= next_correction_at:
                try:
                    candidate = float(inlet.time_correction(timeout=float(inlet_timeout_s)))
                    if np.isfinite(candidate):
                        correction = candidate
                        correction_history.append({"elapsed_capture_s": time.monotonic() - started,
                                                   "correction_s": correction})
                except Exception:
                    pass  # retain the most recent valid correction
                next_correction_at = time.monotonic() + 5.0
            samples, stamps = inlet.pull_chunk(timeout=0.1, max_samples=4096)
            if not stamps:
                continue
            values = np.asarray(samples)
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            if values.ndim != 2 or values.shape[1] <= int(channel_index):
                raise RuntimeError("LSL stream returned an unexpected sample shape.")
            selected = np.asarray(values[:, int(channel_index)], dtype=float)
            times = np.asarray(stamps, dtype=float)
            good = np.isfinite(selected) & np.isfinite(times)
            if good.any():
                corrected_times = times[good] + correction
                valid_corrected = np.isfinite(corrected_times)
                if valid_corrected.any():
                    remaining = int(max_samples) - total_samples
                    selected_chunk = selected[good][valid_corrected][:remaining]
                    time_chunk = corrected_times[valid_corrected][:remaining]
                    if len(selected_chunk):
                        samples_out.append(selected_chunk)
                        timestamps_out.append(time_chunk)
                        total_samples += int(len(selected_chunk))
            if total_samples >= int(max_samples):
                break
    finally:
        try:
            inlet.close_stream()
        except Exception:
            pass
    if not samples_out:
        raise RuntimeError("No numeric LSL samples were received during the capture interval.")
    values = np.concatenate(samples_out)
    timestamps = np.concatenate(timestamps_out)
    original_order_violations = int(np.sum(np.diff(timestamps) <= 0))
    order = np.argsort(timestamps, kind="stable")
    values, timestamps = values[order], timestamps[order]
    # Resolve repeated timestamps deterministically by averaging the repeated samples.
    unique, inverse = np.unique(timestamps, return_inverse=True)
    if len(unique) != len(timestamps):
        sums = np.bincount(inverse, weights=values)
        counts = np.bincount(inverse)
        timestamps, values = unique, sums / counts
    if len(timestamps) < 5 or np.any(np.diff(timestamps) <= 0):
        raise RuntimeError("Captured LSL timestamps are insufficient or not strictly increasing.")
    nominal = float(_safe_call(info, "nominal_srate", 0.0) or 0.0)
    intervals = np.diff(timestamps)
    median_dt = float(np.median(intervals))
    interval_cv = float(np.std(intervals) / median_dt) if median_dt > 0 else float("nan")
    large_gap_count = int(np.sum(intervals > 1.8 * median_dt)) if median_dt > 0 else 0
    resolved_uid = _safe_call(info, "uid", stream_uid or "")
    resolved_source_id = _safe_call(info, "source_id", source_id or "")
    correction_successful = bool(correction_history) if clock_correct else False
    return {
        "timestamps_s": timestamps,
        "values": values,
        "metadata": {
            "stream_name": _safe_call(info, "name", "unnamed"),
            "stream_type": _safe_call(info, "type", ""),
            "source_id": resolved_source_id,
            "stream_uid": resolved_uid,
            "lookup_property": lookup_property,
            "lookup_value": lookup_value,
            "channel_index": int(channel_index),
            "channel_count": channels,
            "nominal_sample_rate_hz": nominal,
            "observed_median_sample_rate_hz": 1.0 / median_dt if median_dt > 0 else None,
            "clock_correction_s_last": correction,
            "clock_correction_successful": correction_successful,
            "clock_correction_history": correction_history,
            "clock_correction_refresh_interval_s": 5.0 if clock_correct else None,
            "clock_correction_application": "latest finite correction applied per received chunk" if clock_correct else "no correction applied",
            "timestamps_clock_domain": "receiver LSL clock" if correction_successful else "outlet LSL clock",
            "timestamps_reordered_count": original_order_violations,
            "captured_samples": int(len(values)),
            "capture_duration_s": float(timestamps[-1] - timestamps[0]),
            "timestamp_interval_cv": interval_cv,
            "large_gap_count_over_1_8x_median_interval": large_gap_count,
        },
    }


def signal_to_csv_bytes(timestamps_s: np.ndarray, values: np.ndarray) -> bytes:
    times = np.asarray(timestamps_s, dtype=float).reshape(-1)
    vals = np.asarray(values, dtype=float).reshape(-1)
    if len(times) != len(vals) or len(times) < 5:
        raise ValueError("Timestamp and signal arrays must have matching lengths and at least five samples.")
    if not np.isfinite(times).all() or not np.isfinite(vals).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Signal timestamps must be finite, strictly increasing and paired with finite values.")
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["time_s", "value"])
    writer.writerows(zip(times.tolist(), vals.tolist()))
    return output.getvalue().encode("utf-8")
