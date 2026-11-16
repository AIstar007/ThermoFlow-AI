"""Protocol-aware timebase and phase segmentation for thermal recordings."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ProtocolTimeline:
    times_s: np.ndarray
    phase_labels: np.ndarray
    phase_spans: list[dict]
    derivative_breaks: tuple[int, ...]
    time_axis_label: str
    warnings: tuple[str, ...]
    metadata: dict


def validate_timestamps(times_s: np.ndarray, n_frames: int) -> np.ndarray:
    values = np.asarray(times_s, dtype=float).reshape(-1)
    if values.shape != (n_frames,):
        raise ValueError(f"Expected {n_frames} timestamps, received {len(values)}")
    if not np.isfinite(values).all():
        raise ValueError("Timestamp values must all be finite")
    if np.any(np.diff(values) <= 0):
        raise ValueError("Timestamps must be strictly increasing with no duplicates")
    return values


def _spans(times: np.ndarray, labels: np.ndarray) -> list[dict]:
    spans: list[dict] = []
    start = 0
    for idx in range(1, len(labels) + 1):
        if idx == len(labels) or labels[idx] != labels[start]:
            spans.append({
                "label": str(labels[start]),
                "start_s": float(times[start]),
                "end_s": float(times[idx - 1]),
                "start_frame": int(start),
                "end_frame": int(idx - 1),
            })
            start = idx
    return spans


def build_protocol_timeline(
    n_frames: int,
    fps: float,
    baseline_frames: int,
    mode: str = "baseline_recovery_unrecorded_challenge",
    timestamps_s: np.ndarray | None = None,
    challenge_start_frame: int | None = None,
    challenge_end_frame: int | None = None,
    challenge_gap_seconds: float | None = None,
    gap_duration_confirmed: bool = False,
    timestamps_include_gap: bool = False,
) -> ProtocolTimeline:
    """Build phase labels and a frame timebase.

    In baseline+recovery mode, the challenge is not present in the frame array.
    A derivative break is inserted at the boundary so dT/dt never spans the
    unobserved interval. If the duration is unknown, the displayed time axis is
    explicitly labelled as recorded time with the gap omitted.
    """
    n = int(n_frames)
    base = int(baseline_frames)
    if n < 3:
        raise ValueError("At least three frames are required")
    if not (2 <= base < n):
        raise ValueError("baseline_frames must be >= 2 and < n_frames")
    if fps <= 0:
        raise ValueError("fps must be positive")

    warnings: list[str] = []
    if timestamps_s is not None:
        times = validate_timestamps(timestamps_s, n)
        time_axis_label = "Acquisition timestamp (s)"
        dt = np.diff(times)
        nominal = float(np.median(dt))
        if nominal > 0 and np.max(dt) > 1.8 * nominal:
            warnings.append("The timestamp vector contains one or more large gaps. Their cause is unverified unless independently documented.")
    else:
        times = np.arange(n, dtype=float) / float(fps)
        time_axis_label = "Recorded time (s)"

    labels = np.full(n, "Recorded sequence", dtype=object)
    breaks: tuple[int, ...] = ()
    metadata: dict = {"mode": mode, "baseline_frames": base, "timestamps_supplied": timestamps_s is not None}

    if mode == "baseline_recovery_unrecorded_challenge":
        labels[:base] = "Baseline"
        labels[base:] = "Rewarming · challenge not recorded"
        breaks = (base,)
        metadata["challenge_interval_recorded"] = False
        metadata["recovery_start_frame"] = base

        if timestamps_s is None:
            if gap_duration_confirmed:
                if challenge_gap_seconds is None or challenge_gap_seconds < 0:
                    raise ValueError("A confirmed challenge gap requires a non-negative duration")
                offset = float(challenge_gap_seconds)
                if offset > 0:
                    times[base:] += offset
                time_axis_label = "Elapsed time including documented unrecorded challenge (s)"
                metadata["unrecorded_challenge_gap_s"] = offset
                metadata["gap_duration_confirmed"] = True
            else:
                time_axis_label = "Recorded time (s; unrecorded challenge gap omitted)"
                metadata["unrecorded_challenge_gap_s"] = None
                metadata["gap_duration_confirmed"] = False
                warnings.append("Challenge duration is unknown: the time axis omits the unrecorded interval. Derivatives are split at the baseline/rewarming boundary.")
        else:
            metadata["gap_duration_confirmed"] = bool(timestamps_include_gap)
            if not timestamps_include_gap:
                time_axis_label = "Acquisition timestamp (s; gap not confirmed as challenge duration)"
                warnings.append("Uploaded timestamps are preserved, but the interval between baseline and rewarming is not labelled as the challenge unless you independently confirmed that interpretation.")
            else:
                time_axis_label = "Acquisition timestamp (s; documented challenge gap included)"
                metadata["unrecorded_challenge_gap_s"] = float(times[base] - times[base - 1] - 1.0 / fps)
        if np.any(np.diff(times) <= 0):
            raise ValueError("The resulting timebase is not strictly increasing. Check FPS, timestamps, and challenge-gap duration.")

    elif mode == "continuous_challenge_recovery":
        start = base if challenge_start_frame is None else int(challenge_start_frame)
        end = min(n - 1, start + 1) if challenge_end_frame is None else int(challenge_end_frame)
        if not (base <= start < end < n):
            raise ValueError("For a recorded challenge, require baseline_frames <= challenge_start < challenge_end < n_frames")
        labels[:start] = "Baseline"
        labels[start:end] = "Recorded challenge"
        labels[end:] = "Recovery"
        metadata.update({"challenge_interval_recorded": True, "challenge_start_frame": start,
                         "challenge_end_frame": end, "recovery_start_frame": end})
        time_axis_label = "Acquisition timestamp (s)" if timestamps_s is not None else "Recorded time (s)"
    else:
        raise ValueError(f"Unsupported protocol mode: {mode}")

    if len(times) > 1:
        dt = np.diff(times)
        nominal = float(np.median(dt))
        jitter = float(np.std(dt) / nominal) if nominal > 0 else float("nan")
        metadata["median_frame_interval_s"] = nominal
        metadata["effective_fps"] = float(1.0 / nominal) if nominal > 0 else None
        metadata["frame_interval_cv"] = jitter
        metadata["max_frame_interval_s"] = float(np.max(dt))
        metadata["min_frame_interval_s"] = float(np.min(dt))
        if jitter > 0.05:
            warnings.append(f"Frame interval variability is {jitter:.1%} (coefficient of variation); verify timestamps and dropped-frame handling.")
    return ProtocolTimeline(times, labels, _spans(times, labels), breaks, time_axis_label,
                             tuple(dict.fromkeys(warnings)), metadata)
