import numpy as np
import pytest

from thermoprop.protocol import build_protocol_timeline, validate_timestamps
from thermoprop.roi import validate_regions, rasterize_regions, overlap_report
from thermoprop.quality import compute_quality_report
from thermoprop.analysis import analyze_sequence
from thermoprop.synthetic import make_propagation_sequence


def sample_regions():
    return [
        {"id": "r1", "name": "Left cheek", "color": "#4de2c5",
         "points": [[1, 1], [8, 1], [8, 8], [1, 8]]},
        {"id": "r2", "name": "Right cheek", "color": "#ffb45c",
         "points": [[10, 1], [18, 1], [18, 8], [10, 8]]},
    ]


def test_protocol_unrecorded_challenge_marks_derivative_break_and_unknown_gap():
    timeline = build_protocol_timeline(810, fps=3, baseline_frames=90,
        mode="baseline_recovery_unrecorded_challenge")
    assert timeline.derivative_breaks == (90,)
    assert timeline.phase_labels[0] == "Baseline"
    assert "challenge not recorded" in timeline.phase_labels[90]
    assert timeline.metadata["gap_duration_confirmed"] is False
    assert any("omits the unrecorded interval" in warning for warning in timeline.warnings)


def test_protocol_can_insert_confirmed_unrecorded_gap():
    timeline = build_protocol_timeline(12, fps=2, baseline_frames=5,
        mode="baseline_recovery_unrecorded_challenge",
        challenge_gap_seconds=120.0, gap_duration_confirmed=True)
    assert timeline.times_s[5] - timeline.times_s[4] == pytest.approx(120.5)
    assert timeline.derivative_breaks == (5,)
    assert "including documented" in timeline.time_axis_label


def test_uploaded_timestamps_are_preserved_and_gap_is_not_assumed():
    times = np.r_[np.arange(5) / 2, 200 + np.arange(7) / 2]
    timeline = build_protocol_timeline(12, 2, 5, timestamps_s=times)
    assert np.allclose(timeline.times_s, times)
    assert timeline.metadata["gap_duration_confirmed"] is False
    assert any("not labelled as the challenge" in w for w in timeline.warnings)


def test_confirmed_uploaded_gap_is_marked_in_metadata():
    times = np.r_[np.arange(5) / 2, 200 + np.arange(7) / 2]
    timeline = build_protocol_timeline(12, 2, 5, timestamps_s=times,
                                      timestamps_include_gap=True)
    assert timeline.metadata["gap_duration_confirmed"] is True
    assert timeline.metadata["unrecorded_challenge_gap_s"] == pytest.approx(197.5)


def test_timestamp_validation_rejects_nonmonotonic_values():
    with pytest.raises(ValueError):
        validate_timestamps(np.array([0, 1, 1, 3]), 4)


def test_polygon_roi_rasterization_and_overlap_report():
    regions = validate_regions(sample_regions(), (20, 20))
    masks = rasterize_regions(regions, (20, 20))
    assert masks["r1"].sum() > 0
    assert masks["r2"].sum() > 0
    assert overlap_report(masks)["overlap_pixels"] == 0


def test_polygon_roi_rejects_degenerate_polygon():
    bad = [{"id": "flat", "name": "Flat", "points": [[1, 1], [2, 2], [3, 3]]}]
    with pytest.raises(ValueError):
        validate_regions(bad, (10, 10))


def test_roi_metric_extraction_uses_shared_full_result():
    from thermoprop.roi import roi_metrics_from_result
    frames, _ = make_propagation_sequence(n_frames=24, height=20, width=20)
    result = analyze_sequence(frames, fps=2, baseline_frames=4, change_threshold=.2)
    mask = np.zeros((20, 20), bool)
    mask[2:10, 2:10] = True
    metrics = roi_metrics_from_result(result, mask, "test")
    assert len(metrics) == 24
    assert set(["roi", "median", "mean_delta_from_baseline", "mean_dT_dt"]).issubset(metrics.columns)
    assert (metrics.roi == "test").all()


def test_quality_report_flags_unverified_timing_variability():
    frames = np.ones((8, 10, 10), dtype=float)
    times = np.array([0, .5, 1, 1.5, 4, 4.5, 5, 5.5])
    roi = np.ones((10, 10), dtype=bool)
    report = compute_quality_report(frames, times, 4, roi)
    assert report["frame_interval_cv"] > .05
    assert any("timestamp" in w.lower() for w in report["warnings"])


def test_continuous_challenge_protocol_labels_three_observed_phases():
    timeline = build_protocol_timeline(30, fps=2, baseline_frames=6,
        mode="continuous_challenge_recovery", challenge_start_frame=6, challenge_end_frame=14)
    assert timeline.phase_labels[0] == "Baseline"
    assert timeline.phase_labels[6] == "Recorded challenge"
    assert timeline.phase_labels[14] == "Recovery"
    assert timeline.derivative_breaks == ()
    assert len(timeline.phase_spans) == 3


def test_overlap_report_detects_shared_pixels():
    regs = [
        {"id":"a","name":"A","points":[[1,1],[10,1],[10,10],[1,10]]},
        {"id":"b","name":"B","points":[[5,5],[14,5],[14,14],[5,14]]},
    ]
    masks = rasterize_regions(regs, (20,20))
    report = overlap_report(masks)
    assert report["overlap_pixels"] > 0
    assert report["overlap_fraction_of_union"] > 0


def test_components_v2_state_list_is_not_dropped_on_rerun():
    from types import SimpleNamespace
    from thermoprop.roi_editor import extract_component_regions
    updated = sample_regions()
    assert extract_component_regions(updated, []) == updated
    assert extract_component_regions({"regions": updated}, []) == updated
    assert extract_component_regions(SimpleNamespace(regions=updated), []) == updated
    assert extract_component_regions(None, updated) == updated


def test_quality_report_uses_outlier_fences_and_large_motion_warning():
    rng = np.random.default_rng(81)
    frames = rng.normal(30.0, 0.3, size=(6, 20, 20))
    frames[5, 10, 10] = 80.0
    frames[4, 0:4, 0:10] = 80.0
    times = np.arange(6, dtype=float) / 2.0
    roi = np.ones((20, 20), dtype=bool)
    shifts = np.zeros((6, 2), dtype=float)
    shifts[4] = [0.0, 8.0]
    report = compute_quality_report(frames, times, 4, roi, shifts)
    assert report["sampled_outlier_pixel_fraction_3_iqr"] > 0
    assert any("3×IQR" in warning for warning in report["warnings"])
    assert any("translation" in warning.lower() for warning in report["warnings"])
