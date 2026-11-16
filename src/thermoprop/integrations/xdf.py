"""Optional XDF import for synchronized LabRecorder physiology streams."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def _first(value, default=""):
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value if value is not None else default


def _channel_labels(info: dict, count: int) -> list[str]:
    try:
        channels = info["desc"][0]["channels"][0]["channel"]
        labels = []
        for index in range(count):
            item = channels[index]
            labels.append(str(_first(item.get("label", [f"channel_{index+1}"]), f"channel_{index+1}")))
        if len(labels) == count:
            return labels
    except Exception:
        pass
    return [f"channel_{index+1}" for index in range(count)]


def load_xdf_streams(payload: bytes, filename: str = "recording.xdf", *,
                     synchronize_clocks: bool = True, dejitter_timestamps: bool = False) -> list[dict[str, Any]]:
    """Load numeric XDF streams using user-selected synchronization settings.

    Clock synchronization is enabled by default. Timestamp de-jitter is disabled
    by default to preserve irregular intervals / frame drops unless the operator
    explicitly chooses to regularize a known fixed-rate signal. The source XDF
    should always be retained unchanged as the authoritative record.
    """
    if not payload:
        raise ValueError("XDF upload is empty.")
    if len(payload) > 2_000_000_000:
        raise ValueError("XDF upload exceeds the 2 GB safety limit; split the recording or use a local workflow.")
    suffix = Path(filename).suffix.lower()
    if suffix != ".xdf":
        raise ValueError("Expected an .xdf file.")
    try:
        import pyxdf
    except ImportError as exc:
        raise RuntimeError("XDF support is optional. Install it with: pip install -e '.[sensors]'") from exc
    with tempfile.TemporaryDirectory(prefix="thermoprop_xdf_") as temp_dir:
        path = Path(temp_dir) / "recording.xdf"
        path.write_bytes(payload)
        streams, header = pyxdf.load_xdf(
            str(path), synchronize_clocks=bool(synchronize_clocks),
            dejitter_timestamps=bool(dejitter_timestamps),
            handle_clock_resets=True, verbose=False,
        )
    output: list[dict[str, Any]] = []
    for index, stream in enumerate(streams):
        info = stream.get("info", {})
        samples = np.asarray(stream.get("time_series", []))
        timestamps = np.asarray(stream.get("time_stamps", []), dtype=float).reshape(-1)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        if samples.ndim != 2 or len(samples) != len(timestamps) or samples.shape[1] < 1:
            continue
        if samples.dtype.kind not in "uif":
            continue
        numeric = samples.astype(float, copy=False)
        channel_count = numeric.shape[1]
        labels = _channel_labels(info, channel_count)
        name = str(_first(info.get("name", []), f"stream_{index+1}"))
        stream_type = str(_first(info.get("type", []), ""))
        source_id = str(_first(info.get("source_id", []), ""))
        good_time = np.isfinite(timestamps)
        if int(good_time.sum()) < 5:
            continue
        timestamps = timestamps[good_time]
        numeric = numeric[good_time]
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError(
                f"XDF stream {name!r} still has non-monotonic timestamps after the selected clock/reset handling. "
                "Inspect the source clock metadata; do not silently sort or regularize this stream."
            )
        intervals = np.diff(timestamps)
        median_interval = float(np.median(intervals)) if len(intervals) else float("nan")
        interval_cv = float(np.std(intervals) / median_interval) if median_interval > 0 else float("nan")
        output.append({
            "stream_index": index,
            "name": name,
            "type": stream_type,
            "source_id": source_id,
            "channel_labels": labels,
            "timestamps_s": timestamps,
            "samples": numeric,
            "nominal_srate": float(_first(info.get("nominal_srate", [0.0]), 0.0) or 0.0),
            "effective_srate": float(_first(info.get("effective_srate", [0.0]), 0.0) or 0.0),
            "sample_count": int(len(timestamps)),
            "channel_count": int(channel_count),
            "channel_finite_fraction": [float(np.isfinite(numeric[:, i]).mean()) for i in range(channel_count)],
            "timestamp_interval_cv": interval_cv,
            "large_timestamp_gap_count_over_1_8x_median": int(np.sum(intervals > 1.8 * median_interval)) if median_interval > 0 else 0,
            "xdf_clock_synchronization_enabled": bool(synchronize_clocks),
            "xdf_dejitter_enabled": bool(dejitter_timestamps),
            "clock_segments": info.get("clock_segments", []),
            "effective_sample_rate_hz": float(_first(info.get("effective_srate", [0.0]), 0.0) or 0.0),
        })
    if not output:
        raise ValueError("No numeric XDF streams with at least five timestamped samples were found.")
    return output
