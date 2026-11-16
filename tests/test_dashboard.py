import numpy as np

from thermoprop.analysis import analyze_sequence
from thermoprop.dashboard import (
    downsample_video,
    make_arrival_figure,
    make_frame_player,
    make_metrics_figure,
    make_named_roi_figure,
    robust_color_limits,
)
from thermoprop.synthetic import make_propagation_sequence


def test_preview_downsamples_large_sequence_without_mutating_source():
    frames = np.arange(140 * 256 * 320, dtype=np.float32).reshape(140, 256, 320)
    before = frames[0, 0, 0]
    preview = downsample_video(frames, max_frames=40, max_side=96)
    assert preview.frames.shape[0] <= 40
    assert max(preview.frames.shape[1:]) <= 96
    assert preview.frame_indices[0] == 0
    assert preview.frame_indices[-1] == 139
    assert frames[0, 0, 0] == before


def test_robust_color_limits_support_symmetric_delta_maps():
    values = np.array([[-2.0, -1.0], [0.5, 4.0]])
    low, high = robust_color_limits(values, symmetric=True)
    assert low == -high
    assert low < 0 < high


def test_animated_frame_player_contains_playback_frames_and_roi():
    frames, _ = make_propagation_sequence(n_frames=32, height=24, width=28)
    times = np.arange(len(frames)) / 2.0
    phases = np.array(["Baseline"] * 8 + ["Recovery"] * (len(frames)-8), dtype=object)
    fig, meta = make_frame_player(frames, times, (3, 4, 25, 20), playback_speed=1.5,
                                  phase_labels=phases)
    assert len(fig.frames) == len(frames)
    assert len(fig.data) == 2  # heatmap and ROI outline
    assert meta["preview_frames"] == len(frames)
    assert fig.layout.sliders[0].steps[-1].label == str(len(frames))
    assert "Baseline" in fig.layout.title.text


def test_arrival_and_roi_charts_are_interactive_plotly_figures():
    frames, _ = make_propagation_sequence(n_frames=40)
    result = analyze_sequence(frames, fps=2, baseline_frames=4, change_threshold=0.3)
    arrival = make_arrival_figure(result)
    metrics = make_metrics_figure(result.metrics, "median", "°C")
    assert len(arrival.data) >= 4
    assert len(metrics.data) == 1
    assert metrics.layout.xaxis.title.text == "Time (s)"


def test_frame_player_supports_multiple_polygon_rois():
    frames, _ = make_propagation_sequence(n_frames=24, height=24, width=28)
    regions = [
        {"name": "ROI A", "color": "#4de2c5", "points": [[1, 1], [10, 1], [10, 10], [1, 10]]},
        {"name": "ROI B", "color": "#ffb45c", "points": [[15, 1], [25, 1], [25, 10], [15, 10]]},
    ]
    fig, meta = make_frame_player(frames, np.arange(24) / 2, None, roi_regions=regions)
    assert len(fig.data) == 3  # heatmap plus two named ROI traces
    assert fig.data[1].name == "ROI A"
    assert fig.data[2].name == "ROI B"


def test_named_roi_plot_adds_protocol_phase_annotations():
    frames, _ = make_propagation_sequence(n_frames=24, height=24, width=24)
    result = analyze_sequence(frames, fps=2, baseline_frames=4)
    regions = {"Full frame": result.metrics.copy(), "Small ROI": result.metrics.copy()}
    fig = make_named_roi_figure(regions, "median", "°C",
        [{"label": "Baseline", "start_s": 0.0, "end_s": 1.5}], "Recorded time (s)")
    assert len(fig.data) == 2
    assert len(fig.layout.shapes) == 1
    assert fig.layout.xaxis.title.text == "Recorded time (s)"


def test_event_progression_player_uses_symmetric_scale_and_phase_label():
    frames = np.linspace(-1.5, 0.7, 12 * 16 * 16, dtype=np.float32).reshape(12, 16, 16)
    phases = np.array(["Baseline"] * 4 + ["Rewarming"] * 8, dtype=object)
    fig, _ = make_frame_player(frames, np.arange(12) / 3.0, view_mode="Event progression",
                              phase_labels=phases)
    heatmap = fig.data[0]
    assert heatmap.zmin == -heatmap.zmax
    assert "Rewarming" in fig.frames[4].layout.title.text
