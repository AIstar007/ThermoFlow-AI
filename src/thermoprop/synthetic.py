"""Deterministic, labelled thermal-like sequences for software verification only."""
from __future__ import annotations

import numpy as np
from scipy.special import expit


def make_propagation_sequence(
    direction: str = "right",
    n_frames: int = 80,
    height: int = 64,
    width: int = 64,
    fps: float = 2.0,
    speed_px_s: float = 4.0,
    amplitude: float = 1.4,
    baseline_c: float = 32.0,
    noise_sd: float = 0.025,
    seed: int = 7,
) -> tuple[np.ndarray, dict]:
    """A planar warm/cold front moving horizontally across a synthetic field."""
    if direction not in {"right", "left"}:
        raise ValueError("direction must be 'right' or 'left'")
    rng = np.random.default_rng(seed)
    times = np.arange(n_frames, dtype=float) / fps
    x = np.arange(width, dtype=np.float32)[None, None, :]
    event_start = 2.0
    front_position = speed_px_s * np.maximum(times - event_start, 0.0)
    if direction == "right":
        activity = expit((front_position[:, None, None] - x) / 1.4)
        signed_amplitude = amplitude
    else:
        x_from_right = (width - 1) - x
        activity = expit((front_position[:, None, None] - x_from_right) / 1.4)
        signed_amplitude = -amplitude
    # A mild fixed vertical gradient avoids unrealistically identical pixels.
    y_gradient = np.linspace(0.15, -0.15, height, dtype=np.float32)[None, :, None]
    video = baseline_c + y_gradient + signed_amplitude * activity
    video = np.broadcast_to(video, (n_frames, height, width)).copy()
    video += rng.normal(0.0, noise_sd, video.shape).astype(np.float32)
    truth = {
        "sequence_type": "synthetic software-validation sequence; not human/clinical data",
        "expected_change": "warming" if direction == "right" else "cooling",
        "expected_direction": direction,
        "expected_speed_px_s": float(speed_px_s),
        "event_start_s": event_start,
        "fps": float(fps),
        "baseline_c": float(baseline_c),
    }
    return video.astype(np.float32), truth
