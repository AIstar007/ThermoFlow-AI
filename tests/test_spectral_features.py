import numpy as np
import pandas as pd

from thermoprop.spectral import welch_spectrum
from thermoprop.feature_engineering import build_participant_feature_table


def test_welch_detects_a_known_thermal_signal_frequency():
    times = np.arange(256, dtype=float) / 2.0
    values = np.sin(2 * np.pi * 0.125 * times)
    freq, power, meta = welch_spectrum(times, values, max_frequency_hz=0.5)
    peak = float(freq[np.argmax(power[1:]) + 1])
    assert abs(peak - 0.125) <= 0.02
    assert meta["nyquist_hz"] == 1.0
    assert meta["method"] == "Welch PSD"


def test_spectrum_does_not_bridge_a_protocol_derivative_break():
    times = np.arange(100, dtype=float)
    values = np.sin(times / 5)
    freq, power, meta = welch_spectrum(times, values, derivative_breaks=(40,), max_frequency_hz=0.2)
    assert meta["segment_start_frame"] in {0, 40}
    assert meta["segment_end_frame_exclusive"] - meta["segment_start_frame"] == 60


def test_participant_feature_export_is_one_row_per_roi_not_per_frame():
    times = np.arange(12, dtype=float)
    data = pd.DataFrame({
        "time_s": times,
        "median": np.linspace(30, 31, 12),
        "std": np.full(12, 0.2),
        "median_event_delta": np.r_[np.full(4, np.nan), np.linspace(0, 1, 8)],
        "mean_dT_dt": np.r_[np.zeros(4), np.linspace(0.1, 0.8, 8)],
        "event_warming_fraction": np.r_[np.full(4, np.nan), np.linspace(0, 1, 8)],
        "event_cooling_fraction": np.zeros(12),
    })
    phases = np.array(["Baseline"]*4 + ["Rewarming · challenge not recorded"]*8, dtype=object)
    out = build_participant_feature_table({"left_hand": data}, {"left_hand": {"valid": True, "tau_s": 12.0}},
        phases, {"ambient_temperature_c": 21.0}, participant_id="P01", session_id="S01")
    assert len(out) == 1
    assert out.iloc[0]["participant_id"] == "P01"
    assert out.iloc[0]["rewarming_tau_s"] == 12.0
    assert np.isnan(out.iloc[0]["reference_perfusion"])
