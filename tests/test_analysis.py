import numpy as np

from thermoprop.analysis import analyze_sequence, rectangular_roi
from thermoprop.bioheat import wu_style_model_index
from thermoprop.recovery import fit_rewarming_time_constant
from thermoprop.synthetic import make_propagation_sequence


def test_warming_front_direction_and_speed():
    frames, truth = make_propagation_sequence(direction="right", seed=11)
    result = analyze_sequence(frames, fps=truth["fps"], baseline_frames=4, change_threshold=0.3)
    warm = result.summary["warming_propagation"]
    assert warm["direction"] == "right"
    assert abs(warm["median_front_speed"] - truth["expected_speed_px_s"]) < 1.8
    assert result.summary["warming_event_fraction"] > 0.7


def test_cooling_front_direction_and_speed():
    frames, truth = make_propagation_sequence(direction="left", seed=12)
    result = analyze_sequence(frames, fps=truth["fps"], baseline_frames=4, change_threshold=0.3)
    cool = result.summary["cooling_propagation"]
    assert cool["direction"] == "left"
    assert abs(cool["median_front_speed"] - truth["expected_speed_px_s"]) < 1.8
    assert result.summary["cooling_event_fraction"] > 0.7


def test_roi_mask_limits_analysis():
    frames, _ = make_propagation_sequence(direction="right", seed=13)
    roi = rectangular_roi(frames.shape[1:], 10, 5, 50, 60)
    result = analyze_sequence(frames, fps=2, baseline_frames=4, roi_mask=roi)
    assert result.roi_mask.sum() == 40 * 55
    assert (result.metrics.roi_pixels == 40 * 55).all()


def test_rewarming_time_constant():
    t = np.linspace(0, 20, 101)
    y = 31.0 - 2.0 * np.exp(-t / 5.0)
    fit = fit_rewarming_time_constant(t, y)
    assert fit["valid"]
    assert abs(fit["tau_s"] - 5.0) < 0.1


def test_paper_model_masks_skin_below_ambient():
    temperatures = np.array([[34.0, 20.0], [35.0, 36.0]], dtype=float)
    estimate = wu_style_model_index(temperatures, ambient_temperature_c=24.0)
    assert np.isfinite(estimate[0, 0])
    assert np.isnan(estimate[0, 1])


def test_threshold_and_timestamp_validation():
    frames, _ = make_propagation_sequence(n_frames=30)
    times = np.arange(len(frames)) * 0.5
    result = analyze_sequence(frames, times_s=times, baseline_frames=4)
    assert np.isclose(result.summary["duration_s"], 14.5)
    try:
        analyze_sequence(frames, times_s=times[::-1], baseline_frames=4)
    except ValueError:
        pass
    else:
        raise AssertionError("Non-increasing timestamps should be rejected")


def test_derivative_is_not_calculated_across_unrecorded_protocol_gap():
    from thermoprop.analysis import analyze_sequence
    frames = np.zeros((12, 16, 16), dtype=np.float32)
    frames[:5] = 30.0
    frames[5:] = 32.0
    # A ten-minute acquisition discontinuity is present, but no dT/dt value should
    # span it when the challenge interval was not recorded.
    times = np.r_[np.arange(5) / 2.0, 300.0 + np.arange(7) / 2.0]
    result = analyze_sequence(frames, times_s=times, baseline_frames=5,
                              change_threshold=0.2, derivative_breaks=(5,))
    assert np.isfinite(result.delta_rate[:5]).all()
    assert np.isfinite(result.delta_rate[5:]).all()
    assert np.allclose(result.delta_rate, 0.0, atol=1e-6)
    assert result.summary["derivative_breaks"] == [5]


def test_event_reference_detects_rewarming_while_below_pre_challenge_baseline():
    # Baseline is warm; the first recorded recovery frame is globally cold. A
    # rightward rewarming front moves toward the pre-challenge temperature but
    # never exceeds it, so baseline-relative positive thresholding would miss it.
    n, h, w, fps = 80, 48, 64, 2.0
    frames = np.full((n, h, w), 34.0, dtype=np.float32)
    x = np.arange(w, dtype=np.float32)[None, :]
    for i in range(4, n):
        t = (i - 4) / fps
        front = 4.0 * t
        warm_fraction = 1.0 / (1.0 + np.exp(-(front - x) / 1.2))
        frames[i] = 30.0 + 3.8 * warm_fraction
    result = analyze_sequence(frames, fps=fps, baseline_frames=4, change_threshold=0.25,
                              front_reference_frame=4, front_reference_frames=2,
                              derivative_breaks=(4,))
    assert np.nanmedian(result.delta[10:]) < 0  # still below the pre-challenge baseline
    assert result.summary["warming_propagation"]["direction"] == "right"
    assert result.summary["warming_event_fraction"] > 0.5
    assert result.summary["front_reference_frame"] == 4
    assert np.isnan(result.event_delta[:4]).all()  # pre-challenge frames are excluded from event progression
    assert "event-relative" in result.summary["front_reference_description"]


def test_temporal_smoothing_does_not_blend_across_protocol_break():
    frames = np.zeros((20, 12, 12), dtype=np.float32)
    frames[:10] = 30.0
    frames[10:] = 35.0
    result = analyze_sequence(frames, fps=1.0, baseline_frames=5, change_threshold=0.2,
                              temporal_smoothing_sigma=2.0, derivative_breaks=(10,))
    # The recovery segment is constant; smoothing must not pull its first values
    # toward the baseline segment across an unobserved stimulus gap.
    assert np.allclose(result.delta[10:, 4, 4], 5.0, atol=1e-5)
    assert np.allclose(result.delta[:10, 4, 4], 0.0, atol=1e-5)
