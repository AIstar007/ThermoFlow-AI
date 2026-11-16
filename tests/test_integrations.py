import io

import numpy as np
import pandas as pd
import pytest

from thermoprop.integrations.cameras import get_profiles, get_profile, adapter_status
from thermoprop.integrations.signals import parse_reference_csv, align_reference_signal, compare_reference_signal
from thermoprop.preprocess import preprocess_sequence
from thermoprop.synthetic import make_propagation_sequence


def test_camera_profiles_cover_expected_vendors_and_generic_import():
    ids = {p.id for p in get_profiles()}
    assert {"generic", "flir", "infratec", "optris", "seek", "hikmicro", "testo", "fluke", "infiray", "gentl", "other"} <= ids
    assert ".seq" in get_profile("flir").native_extensions
    assert ".irb" in get_profile("infratec").native_extensions
    assert ".tif" in get_profile("generic").standard_extensions


def test_camera_profile_status_does_not_claim_native_decode_without_plugin():
    status = adapter_status("hikmicro")
    assert status["profile"] == "HIKMICRO"
    assert isinstance(status["plugin_installed"], bool)
    assert status["native_decode_available"] == status["plugin_installed"]
    assert "universal" in status["offline_sdk"].lower()


def test_reference_csv_parse_and_alignment_without_extrapolation():
    csv = pd.DataFrame({"time_s": np.arange(10, dtype=float), "lsci": np.arange(10, dtype=float)*2}).to_csv(index=False).encode()
    signal = parse_reference_csv(csv, "time_s", "lsci", name="LSCI", modality="lsci", units="a.u.")
    aligned = align_reference_signal(signal, np.array([-1, 0, 1, 5, 9, 10], dtype=float))
    assert np.isnan(aligned[0]) and np.isnan(aligned[-1])
    assert np.allclose(aligned[1:-1], [0, 2, 10, 18])


def test_reference_signal_comparison_reports_descriptive_correlation_and_lag():
    t = np.arange(60, dtype=float)
    thermal = np.sin(t/7)
    ref = np.sin((t-2)/7)
    csv = pd.DataFrame({"t": t, "v": ref}).to_csv(index=False).encode()
    signal = parse_reference_csv(csv, "t", "v", name="Doppler", modality="laser-doppler", units="flux")
    stats, frame = compare_reference_signal(signal, t, thermal, max_lag_s=5)
    assert stats["n_aligned"] == 60
    assert stats["pearson_r"] > 0.8
    assert "descriptive" in stats["interpretation"].lower()
    assert len(frame) == 60


def test_affine_registration_mode_runs_and_records_transforms():
    pytest.importorskip("cv2")
    base = np.zeros((48, 48), dtype=np.float32)
    base[12:30, 15:28] = 1
    frames = np.stack([base, base, base])
    out, meta = preprocess_sequence(frames, spatial_sigma=0.0, register=True,
                                   registration_reference_frames=2, registration_mode="affine_ecc")
    assert out.shape == frames.shape
    assert meta["registration_mode"] == "affine_ecc"
    assert len(meta["affine_transforms_2x3"]) == 3


def test_grouped_ml_training_requires_enough_participants():
    from thermoprop.modeling import train_grouped_reference_model
    table = pd.DataFrame({"participant_id": ["A", "A", "B", "B", "C", "C"],
                          "reference_perfusion": [1, 1.1, 2, 2.1, 3, 3.1],
                          "mean_temperature": [30, 30.1, 31, 31.1, 32, 32.1],
                          "warming_rate": [0.1, 0.2, 0.3, 0.2, 0.5, 0.4]})
    with pytest.raises(ValueError, match="At least 6 distinct"):
        train_grouped_reference_model(table, target_column="reference_perfusion",
                                      group_column="participant_id", feature_columns=["mean_temperature", "warming_rate"])


def test_grouped_ml_training_returns_out_of_fold_metrics():
    pytest.importorskip("sklearn")
    from thermoprop.modeling import train_grouped_reference_model
    rows = []
    for i in range(8):
        for j in range(2):
            x = i + j*0.1
            rows.append({"participant_id": f"P{i}", "reference_perfusion": 2*x+0.4,
                         "mean_temperature": x, "warming_rate": x*0.5 + 0.2})
    table = pd.DataFrame(rows)
    result = train_grouped_reference_model(table, target_column="reference_perfusion",
        group_column="participant_id", model_name="ElasticNet", feature_columns=["mean_temperature", "warming_rate"])
    assert result.metrics["participants"] == 8
    assert result.predictions["oof_prediction"].notna().all()
    assert result.metrics["rmse"] < 2.0
    assert any("participant" in w.lower() for w in result.warnings)


def test_camera_connection_options_are_capability_specific():
    assert "GenTL" in get_profile("gentl").live_connections
    assert "GenTL" not in get_profile("flir").live_connections
    assert any("model-specific" in option for option in get_profile("testo").live_connections)
    assert any("supported models" in option for option in get_profile("hikmicro").live_connections)


def test_camera_export_profiles_do_not_claim_live_driver_without_plugin():
    for profile_id in ("testo", "fluke", "infiray"):
        status = adapter_status(profile_id)
        assert status["native_decode_available"] is False
        assert status["live_probe_available"] is False
        assert "adapter" in status["status_note"].lower()


def test_reference_lag_uses_actual_timestamps_and_respects_protocol_breaks():
    t = np.cumsum(np.r_[0.0, np.tile([0.9, 1.1], 29)])
    thermal = np.sin(t / 8.0)
    delayed_times = t
    delayed_values = np.sin((delayed_times - 2.0) / 8.0)
    csv = pd.DataFrame({"t": delayed_times, "v": delayed_values}).to_csv(index=False).encode()
    signal = parse_reference_csv(csv, "t", "v", name="LSCI", modality="lsci", units="a.u.")
    stats, _ = compare_reference_signal(signal, t, thermal, max_lag_s=4.0, derivative_breaks=[30])
    assert stats["best_lag_s"] == pytest.approx(2.0, abs=0.25)
    assert stats["lag_search_respects_protocol_breaks"] is True
    assert stats["lag_segments_used"] == 2


def test_hikmicro_profile_advertises_only_model_specific_live_path():
    profile = get_profile("hikmicro")
    assert len(profile.live_connections) == 1
    assert "model-specific" in profile.live_connections[0].lower() or "supported models" in profile.live_connections[0].lower()
    status = adapter_status("hikmicro")
    if not status["plugin_installed"]:
        assert status["live_probe_available"] is False
        assert status["live_acquisition_available"] is False


def test_camera_adapter_parameter_schema_is_validated(monkeypatch):
    import thermoprop.integrations.cameras as cameras

    class FakeAdapter:
        def get_parameter_schema(self, connection=None):
            return [
                {"key": "emissivity", "type": "float", "default": 0.98, "min": 0.1, "max": 1.0},
                {"key": "range", "type": "select", "options": ["low", "high"], "default": "high"},
            ]

    monkeypatch.setattr(cameras, "get_adapter", lambda profile_id: FakeAdapter())
    schema = cameras.get_parameter_schema("flir", "USB")
    assert [item["key"] for item in schema] == ["emissivity", "range"]
    assert schema[1]["default"] == "high"


def test_camera_adapter_rejects_duplicate_parameter_schema(monkeypatch):
    import thermoprop.integrations.cameras as cameras

    class FakeAdapter:
        def get_parameter_schema(self, connection=None):
            return [{"key": "gain", "type": "int"}, {"key": "gain", "type": "float"}]

    monkeypatch.setattr(cameras, "get_adapter", lambda profile_id: FakeAdapter())
    with pytest.raises(ValueError, match="duplicate"):
        cameras.get_parameter_schema("flir", "USB")


def test_camera_healthcheck_normalizes_boolean_and_mapping(monkeypatch):
    import thermoprop.integrations.cameras as cameras

    class FakeAdapter:
        def healthcheck(self, device_id=None, **options):
            return {"ok": True, "device_id": device_id}

    monkeypatch.setattr(cameras, "get_adapter", lambda profile_id: FakeAdapter())
    result = cameras.healthcheck_camera("flir", "CAM-1", connection="USB")
    assert result["ok"] is True
    assert result["device_id"] == "CAM-1"


def test_multi_camera_adapter_sequences_are_validated(monkeypatch):
    import thermoprop.integrations.cameras as cameras

    class FakeAdapter:
        def acquire_multi_sequence(self, connection, device_ids, duration_s, frame_rate_hz, settings):
            frames = np.ones((5, 16, 16), dtype=np.float32)
            return {"recordings": {device_id: {"frames": frames, "timestamps_s": np.arange(5) / 3,
                "metadata": {"serial": device_id}} for device_id in device_ids},
                "metadata": {"synchronization_method": "shared hardware trigger"}}

    monkeypatch.setattr(cameras, "get_adapter", lambda profile_id: FakeAdapter())
    result = cameras.acquire_multiple_camera_sequences("flir", "USB", ["A", "B"], duration_s=2, frame_rate_hz=3)
    assert set(result["recordings"]) == {"A", "B"}
    assert result["recordings"]["A"]["frames"].shape == (5, 16, 16)
    assert result["metadata"]["synchronization_method"] == "shared hardware trigger"
    assert result["metadata"]["synchronization_verified_by_adapter"] is False


def test_multi_camera_adapter_requires_all_requested_device_outputs(monkeypatch):
    import thermoprop.integrations.cameras as cameras

    class FakeAdapter:
        def acquire_multi_sequence(self, connection, device_ids, duration_s, frame_rate_hz, settings):
            return {"recordings": {device_ids[0]: {"frames": np.ones((5, 16, 16))}}}

    monkeypatch.setattr(cameras, "get_adapter", lambda profile_id: FakeAdapter())
    with pytest.raises(ValueError, match="omitted requested devices"):
        cameras.acquire_multiple_camera_sequences("flir", "USB", ["A", "B"])


def test_sensor_signal_csv_serialization_roundtrips():
    from thermoprop.integrations.reference_streams import signal_to_csv_bytes
    times = np.arange(10, dtype=float) * 0.25
    values = np.cos(times)
    payload = signal_to_csv_bytes(times, values)
    signal = parse_reference_csv(payload, "time_s", "value", name="LSCI", modality="lsci", units="a.u.")
    assert np.allclose(signal.timestamps_s, times)
    assert np.allclose(signal.values, values)
    assert signal.quality["estimated_sample_rate_hz"] == pytest.approx(4.0)


def test_reference_signal_quality_records_bad_rows_and_timestamp_repair():
    csv = pd.DataFrame({"t": [0.0, 1.0, 1.0, 2.0, np.nan, 4.0, 3.0],
                        "v": [1.0, 2.0, 4.0, 3.0, 5.0, 5.0, 4.0]}).to_csv(index=False).encode()
    signal = parse_reference_csv(csv, "t", "v", name="Doppler", modality="laser-doppler", units="flux")
    assert signal.quality["invalid_rows_removed"] == 1
    assert signal.quality["duplicate_timestamps_averaged"] == 1
    assert signal.quality["non_monotonic_input_pairs_sorted"] >= 1
    assert np.all(np.diff(signal.timestamps_s) > 0)


def test_xdf_import_rejects_wrong_extension_before_optional_dependency():
    from thermoprop.integrations.xdf import load_xdf_streams
    with pytest.raises(ValueError, match="Expected an .xdf"):
        load_xdf_streams(b"not an xdf", "recording.csv")


def test_tree_shap_explanation_returns_ranked_features_when_available():
    shap = pytest.importorskip("shap")
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from thermoprop.modeling import explain_tree_model

    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["temperature", "warming_rate", "humidity"])
    y = 2.0 * X["temperature"] - 0.5 * X["warming_rate"] + rng.normal(0, 0.05, len(X))
    fitted = Pipeline([("imputer", SimpleImputer(strategy="median")),
                       ("model", RandomForestRegressor(n_estimators=20, random_state=3, min_samples_leaf=1))])
    fitted.fit(X, y)
    importance = explain_tree_model(fitted, X, max_samples=30)
    assert list(importance.columns) == ["feature", "mean_absolute_shap", "median_absolute_shap"]
    assert importance.iloc[0]["feature"] == "temperature"
    assert np.all(np.diff(importance["mean_absolute_shap"].to_numpy()) <= 1e-12)


def test_multi_camera_adapter_must_declare_synchronization_method(monkeypatch):
    import thermoprop.integrations.cameras as cameras

    class FakeAdapter:
        def acquire_multi_sequence(self, connection, device_ids, duration_s, frame_rate_hz, settings):
            frames = np.ones((5, 16, 16), dtype=np.float32)
            return {"recordings": {device_id: {"frames": frames, "timestamps_s": np.arange(5) / 3}
                                   for device_id in device_ids},
                    "metadata": {}}

    monkeypatch.setattr(cameras, "get_adapter", lambda profile_id: FakeAdapter())
    with pytest.raises(ValueError, match="synchronization_method"):
        cameras.acquire_multiple_camera_sequences("flir", "USB", ["A", "B"])


def test_multi_camera_adapter_requires_timestamps_for_each_device(monkeypatch):
    import thermoprop.integrations.cameras as cameras

    class FakeAdapter:
        def acquire_multi_sequence(self, connection, device_ids, duration_s, frame_rate_hz, settings):
            frames = np.ones((5, 16, 16), dtype=np.float32)
            return {"recordings": {
                "A": {"frames": frames, "timestamps_s": np.arange(5) / 3},
                "B": {"frames": frames, "timestamps_s": None},
            }, "metadata": {"synchronization_method": "shared trigger"}}

    monkeypatch.setattr(cameras, "get_adapter", lambda profile_id: FakeAdapter())
    with pytest.raises(ValueError, match="needs timestamps"):
        cameras.acquire_multiple_camera_sequences("flir", "USB", ["A", "B"])


def test_camera_adapter_factory_returning_none_is_not_reported_as_loaded(monkeypatch):
    import thermoprop.integrations.cameras as cameras

    class Entry:
        name = "empty-adapter"

    monkeypatch.setattr(cameras, "discover_camera_adapters", lambda: {"flir": Entry()})
    monkeypatch.setattr(cameras, "get_adapter", lambda profile_id: None)
    status = cameras.adapter_status("flir")
    assert status["plugin_installed"] is True
    assert status["adapter_loaded"] is False
    assert status["plugin_load_error"] == "Adapter entry point factory returned None."
    assert status["live_acquisition_available"] is False


def test_camera_parameter_schema_rejects_default_outside_device_bounds(monkeypatch):
    import thermoprop.integrations.cameras as cameras

    class FakeAdapter:
        def get_parameter_schema(self, connection=None):
            return [{"key": "emissivity", "type": "float", "default": 1.2,
                     "min": 0.1, "max": 1.0, "step": 0.01}]

    monkeypatch.setattr(cameras, "get_adapter", lambda profile_id: FakeAdapter())
    with pytest.raises(ValueError, match="outside valid bounds"):
        cameras.get_parameter_schema("flir", "USB")


def test_xdf_loader_passes_separate_clock_sync_and_dejitter_choices(monkeypatch):
    import sys
    from types import SimpleNamespace
    from thermoprop.integrations.xdf import load_xdf_streams

    observed = {}
    stream = {
        "info": {"name": ["LSCI"], "type": ["Physiology"], "source_id": ["lsci-1"],
                 "nominal_srate": ["10"], "desc": []},
        "time_series": np.column_stack([np.arange(10, dtype=float), np.arange(10, dtype=float) * 2]),
        "time_stamps": np.arange(10, dtype=float) / 10,
    }

    def fake_load_xdf(path, **kwargs):
        observed.update(kwargs)
        return [stream], {"info": "header"}

    monkeypatch.setitem(sys.modules, "pyxdf", SimpleNamespace(load_xdf=fake_load_xdf))
    result = load_xdf_streams(b"fake-xdf-content", "test.xdf", synchronize_clocks=True,
                              dejitter_timestamps=False)
    assert len(result) == 1
    assert result[0]["channel_count"] == 2
    assert result[0]["xdf_clock_synchronization_enabled"] is True
    assert result[0]["xdf_dejitter_enabled"] is False
    assert observed["synchronize_clocks"] is True
    assert observed["dejitter_timestamps"] is False
    assert observed["handle_clock_resets"] is True


def test_xdf_loader_rejects_nonmonotonic_stream_timestamps_after_clock_handling(monkeypatch):
    import sys
    from types import SimpleNamespace
    from thermoprop.integrations.xdf import load_xdf_streams

    stream = {
        "info": {"name": ["ECG"], "type": ["Physiology"], "source_id": ["ecg-1"],
                 "nominal_srate": ["10"], "desc": []},
        "time_series": np.arange(10, dtype=float).reshape(-1, 1),
        "time_stamps": np.array([0, 1, 2, 3, 4, 5, 6, 7, 6.5, 9], dtype=float),
    }
    monkeypatch.setitem(sys.modules, "pyxdf", SimpleNamespace(load_xdf=lambda *a, **k: ([stream], {})))
    with pytest.raises(ValueError, match="non-monotonic"):
        load_xdf_streams(b"fake-xdf-content", "test.xdf")


def test_lsl_capture_can_resolve_unique_stream_uid_and_reports_clock_metadata(monkeypatch):
    import sys
    import time
    from types import SimpleNamespace
    from thermoprop.integrations.reference_streams import capture_lsl_channel

    class FakeInfo:
        def channel_count(self): return 2
        def nominal_srate(self): return 10.0
        def uid(self): return "uid-123"
        def source_id(self): return "source-123"
        def name(self): return "Reference sensor"
        def type(self): return "Physiology"

    class FakeInlet:
        def __init__(self, info, max_buflen=60): self.sent = False
        def open_stream(self, timeout=5): return None
        def time_correction(self, timeout=5): return 0.25
        def pull_chunk(self, timeout=0.1, max_samples=4096):
            if not self.sent:
                self.sent = True
                return [[float(i), float(i * 2)] for i in range(10)], [i / 10 for i in range(10)]
            time.sleep(0.01)
            return [], []
        def close_stream(self): return None

    observed = {}
    fake_pylsl = SimpleNamespace(
        resolve_byprop=lambda prop, value, **kwargs: (observed.update({"property": prop, "value": value}) or [FakeInfo()]),
        StreamInlet=FakeInlet,
    )
    monkeypatch.setitem(sys.modules, "pylsl", fake_pylsl)
    result = capture_lsl_channel(stream_uid="uid-123", channel_index=1, duration_s=0.1, clock_correct=True)
    assert observed == {"property": "uid", "value": "uid-123"}
    assert np.allclose(result["values"], np.arange(10) * 2)
    assert np.allclose(result["timestamps_s"], np.arange(10) / 10 + 0.25)
    assert result["metadata"]["stream_uid"] == "uid-123"
    assert result["metadata"]["clock_correction_successful"] is True
    assert result["metadata"]["timestamps_clock_domain"] == "receiver LSL clock"


def test_vendor_native_loader_passes_selected_camera_settings(monkeypatch):
    import thermoprop.integrations.cameras as cameras

    class FakeAdapter:
        def load_recording(self, payload, filename, settings=None, **options):
            assert payload == b"native"
            assert filename == "recording.fake"
            return {"frames": np.ones((5, 16, 16), dtype=np.float32),
                    "timestamps_s": np.arange(5, dtype=float),
                    "metadata": {"settings_received": dict(settings or {})}}

    monkeypatch.setattr(cameras, "get_adapter", lambda profile_id: FakeAdapter())
    result = cameras.load_vendor_recording("flir", b"native", "recording.fake",
                                          settings={"emissivity": 0.97, "range": "high"})
    assert result["metadata"]["settings_received"] == {"emissivity": 0.97, "range": "high"}


def test_camera_parameter_schema_rejects_nonfinite_range_and_duplicate_select_options(monkeypatch):
    import thermoprop.integrations.cameras as cameras

    class NonfiniteAdapter:
        def get_parameter_schema(self, connection=None):
            return [{"key": "gain", "type": "float", "default": 1.0,
                     "min": float("nan"), "max": 2.0, "step": 0.1}]

    monkeypatch.setattr(cameras, "get_adapter", lambda profile_id: NonfiniteAdapter())
    with pytest.raises(ValueError, match="finite"):
        cameras.get_parameter_schema("flir", "USB")

    class DuplicateSelectAdapter:
        def get_parameter_schema(self, connection=None):
            return [{"key": "range", "type": "select", "options": ["low", "high", "low"]}]

    monkeypatch.setattr(cameras, "get_adapter", lambda profile_id: DuplicateSelectAdapter())
    with pytest.raises(ValueError, match="duplicate"):
        cameras.get_parameter_schema("flir", "USB")


def test_camera_boolean_parameter_schema_rejects_string_default(monkeypatch):
    import thermoprop.integrations.cameras as cameras

    class FakeAdapter:
        def get_parameter_schema(self, connection=None):
            return [{"key": "nuc", "type": "bool", "default": "false"}]

    monkeypatch.setattr(cameras, "get_adapter", lambda profile_id: FakeAdapter())
    with pytest.raises(ValueError, match="boolean default"):
        cameras.get_parameter_schema("flir", "USB")


def test_recorded_file_is_not_treated_as_live_camera_connection(monkeypatch):
    import thermoprop.integrations.cameras as cameras

    class FakeAdapter:
        def acquire_sequence(self, **kwargs):
            raise AssertionError("recorded-file workflow must not invoke live acquisition")

    monkeypatch.setattr(cameras, "get_adapter", lambda profile_id: FakeAdapter())
    with pytest.raises(ValueError, match="not a live device connection"):
        cameras.acquire_camera_sequence("flir", "Recorded file", duration_s=1.0, frame_rate_hz=1.0)


def test_grouped_ml_training_reports_optional_dependency_after_data_checks(monkeypatch):
    import builtins
    from thermoprop.modeling import train_grouped_reference_model

    records = []
    for i in range(6):
        for j in range(2):
            value = float(i) + 0.1 * j
            records.append({
                "participant_id": f"P{i}",
                "reference_perfusion": 1.0 + value,
                "mean_temperature": 30.0 + value,
                "warming_rate": 0.1 + value * 0.05,
            })
    table = pd.DataFrame(records)
    real_import = builtins.__import__

    def without_sklearn(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ModuleNotFoundError("No module named 'sklearn'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sklearn)
    with pytest.raises(RuntimeError, match=r"pip install -e '\.\[ml\]'" ):
        train_grouped_reference_model(
            table,
            target_column="reference_perfusion",
            group_column="participant_id",
            feature_columns=["mean_temperature", "warming_rate"],
        )


def test_camera_catalog_covers_major_vendor_and_transport_families():
    from thermoprop.integrations.cameras import get_profiles
    ids = {profile.id for profile in get_profiles()}
    assert {
        "generic", "flir", "infratec", "optris", "seek", "hikmicro", "hikvision",
        "testo", "fluke", "infiray", "guide", "workswell", "xenics", "telops",
        "sierra_olympic", "axis", "mobotix", "basler", "allied_vision", "ids", "lucid",
        "teledyne_dalsa", "ximea", "raptor_photonics", "nit", "smartphone_core",
        "thermal_array", "gentl", "http_matrix", "other"
    }.issubset(ids)


def test_http_matrix_builtin_adapter_reports_capabilities_and_password_schema():
    from thermoprop.integrations.cameras import adapter_status, get_parameter_schema
    status = adapter_status("http_matrix")
    assert status["adapter_loaded"] is True
    assert status["adapter_origin"] == "built-in"
    assert status["native_decode_available"] is True
    assert status["live_acquisition_available"] is True
    schema = get_parameter_schema("http_matrix", "HTTP radiometric matrix API")
    fields = {field["key"]: field for field in schema}
    assert fields["password"]["type"] == "password"
    assert fields["bearer_token"]["type"] == "password"
    assert fields["frames_key"]["default"] == "frames"


def test_http_matrix_json_decode_preserves_values_timestamps_and_provenance():
    import json
    from thermoprop.integrations.cameras import load_vendor_recording
    frames = np.arange(3 * 16 * 16, dtype=np.uint16).reshape(3, 16, 16)
    payload = json.dumps({"frames": frames.tolist(), "timestamps_s": [0.0, 0.5, 1.0],
                          "metadata": {"model": "mock-radiometric-camera"}}).encode()
    result = load_vendor_recording("http_matrix", payload, "recording.json", settings={
        "units": "°C", "calibration_confirmed": True,
        "frames_key": "frames", "timestamps_key": "timestamps_s"
    })
    assert result["frames"].shape == (3, 16, 16)
    assert np.allclose(result["frames"], frames)
    assert np.allclose(result["timestamps_s"], [0.0, 0.5, 1.0])
    assert result["metadata"]["units"] == "°C"
    assert result["metadata"]["calibration_status"] == "operator_confirmed"
    assert result["metadata"]["model"] == "mock-radiometric-camera"


def test_http_matrix_does_not_allow_unconfirmed_celsius():
    import json
    from thermoprop.integrations.cameras import load_vendor_recording
    payload = json.dumps({"frames": np.zeros((3, 16, 16)).tolist()}).encode()
    with pytest.raises(ValueError, match="not confirmed"):
        load_vendor_recording("http_matrix", payload, "capture.json", settings={
            "units": "°C", "calibration_confirmed": False
        })


def test_http_matrix_rejects_non_http_url_and_embedded_credentials():
    from thermoprop.integrations.builtin_camera_adapters import HttpRadiometricMatrixAdapter
    adapter = HttpRadiometricMatrixAdapter()
    with pytest.raises(ValueError, match="valid http"):
        adapter._url({"endpoint_url": "ftp://example.com/capture"})
    with pytest.raises(ValueError, match="Do not embed credentials"):
        adapter._url({"endpoint_url": "https://user:pass@example.com/matrix"})


def test_gentl_builtin_probe_uses_cti_and_returns_device_metadata(monkeypatch, tmp_path):
    import sys
    import types
    from thermoprop.integrations.builtin_camera_adapters import GenTLAdapter

    cti = tmp_path / "mock.cti"
    cti.write_bytes(b"mock producer placeholder")

    class DeviceInfo:
        id_ = "DEV-1"
        serial_number = "SER-1"
        vendor = "ExampleVendor"
        model = "ThermalModel"
        tl_type = "GigE"

    class FakeHarvester:
        def __init__(self):
            self.device_info_list = []
            self.added = None
        def add_file(self, path, **kwargs):
            self.added = path
        def update(self):
            self.device_info_list = [DeviceInfo()]
        def reset(self):
            pass

    pkg = types.ModuleType("harvesters")
    core = types.ModuleType("harvesters.core")
    core.Harvester = FakeHarvester
    pkg.core = core
    monkeypatch.setitem(sys.modules, "harvesters", pkg)
    monkeypatch.setitem(sys.modules, "harvesters.core", core)

    devices = GenTLAdapter().probe("GenTL", settings={"cti_file": str(cti)})
    assert len(devices) == 1
    assert devices[0]["device_id"] == "SER-1"
    assert devices[0]["model"] == "ThermalModel"
    assert devices[0]["tl_type"] == "GigE"


def test_gentl_builtin_capture_returns_scalar_counts_and_camera_timestamps(monkeypatch, tmp_path):
    import sys
    import types
    from thermoprop.integrations.builtin_camera_adapters import GenTLAdapter

    cti = tmp_path / "mock.cti"
    cti.write_bytes(b"mock producer placeholder")

    class DeviceInfo:
        id_ = "DEV-1"
        serial_number = "SER-1"
        vendor = "ExampleVendor"
        model = "ThermalModel"
        tl_type = "GigE"

    class Component:
        height = 16
        width = 16
        data_format = "Mono16"
        data = np.arange(256, dtype=np.uint16).reshape(16, 16)

    class Payload:
        components = [Component()]

    class Buffer:
        def __init__(self, idx):
            self.payload = Payload()
            self.timestamp_ns = int(idx * 100_000_000)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    class FakeIA:
        def __init__(self):
            self.i = 0
        def start(self):
            pass
        def fetch(self, timeout=0):
            self.i += 1
            return Buffer(self.i)
        def is_acquiring(self):
            return True
        def stop(self):
            pass
        def destroy(self):
            pass

    class FakeHarvester:
        def __init__(self):
            self.device_info_list = [DeviceInfo()]
        def add_file(self, path, **kwargs):
            self.path = path
        def update(self):
            pass
        def create(self, search_key=None):
            return FakeIA()
        def reset(self):
            pass

    pkg = types.ModuleType("harvesters")
    core = types.ModuleType("harvesters.core")
    core.Harvester = FakeHarvester
    pkg.core = core
    monkeypatch.setitem(sys.modules, "harvesters", pkg)
    monkeypatch.setitem(sys.modules, "harvesters.core", core)

    result = GenTLAdapter().acquire_sequence("GenTL", device_id="SER-1", duration_s=0.5,
        frame_rate_hz=4, settings={"cti_file": str(cti), "frame_timeout_s": 0.2})
    assert result["frames"].ndim == 3
    assert result["frames"].shape[1:] == (16, 16)
    assert result["metadata"]["units"] == "camera_counts"
    assert result["metadata"]["calibration_status"] == "not_decoded_by_generic_gentl_adapter"
    assert result["timestamps_s"][0] == pytest.approx(0.0)
    assert np.all(np.diff(result["timestamps_s"]) > 0)


def test_camera_architecture_metadata_options_cover_detector_and_output_classes():
    from thermoprop.integrations.cameras import (
        CAMERA_FORM_FACTORS, CAMERA_INTERFACES, DETECTOR_TECHNOLOGIES, SPECTRAL_BANDS,
        DETECTOR_OUTPUT_TYPES, CALIBRATION_STATES,
    )
    assert "UAV / airborne thermal payload" in CAMERA_FORM_FACTORS
    assert "GenICam / GenTL CTI Producer" in CAMERA_INTERFACES
    assert "Camera Link / frame grabber" in CAMERA_INTERFACES
    assert "Uncooled microbolometer" in DETECTOR_TECHNOLOGIES
    assert {"SWIR", "MWIR", "LWIR"}.issubset(set(SPECTRAL_BANDS))
    assert "False-colour RGB / palette image (visual only)" in DETECTOR_OUTPUT_TYPES
    assert "Unknown / not verified" in CALIBRATION_STATES


def test_generic_profile_file_import_does_not_require_camera_adapter():
    from thermoprop.integrations.cameras import adapter_status
    status = adapter_status("generic")
    assert status["standard_extensions"]
    assert status["live_acquisition_available"] is False
    assert status["native_decode_available"] is False


def test_gentl_uses_host_timestamps_consistently_when_device_timestamps_are_partial(monkeypatch, tmp_path):
    import sys
    import types
    from thermoprop.integrations.builtin_camera_adapters import GenTLAdapter

    cti = tmp_path / "mock.cti"
    cti.write_bytes(b"mock")

    class DeviceInfo:
        id_ = "DEV-HOST"
        serial_number = "SER-HOST"
        vendor = "Example"
        model = "NoTimestampThermal"
        tl_type = "USB3"

    class Component:
        height = 16
        width = 16
        data_format = "Mono16"
        data = np.ones((16, 16), dtype=np.uint16)

    class Payload:
        components = [Component()]

    class Buffer:
        def __init__(self, idx):
            self.payload = Payload()
            self.timestamp_ns = 10_000_000 * idx if idx == 1 else None
        def __enter__(self): return self
        def __exit__(self, *args): return False

    class FakeIA:
        def __init__(self): self.i = 0
        def start(self): pass
        def fetch(self, timeout=0):
            self.i += 1
            return Buffer(self.i)
        def is_acquiring(self): return True
        def stop(self): pass
        def destroy(self): pass

    class FakeHarvester:
        def __init__(self): self.device_info_list = [DeviceInfo()]
        def add_file(self, path, **kwargs): pass
        def update(self): pass
        def create(self, search_key=None): return FakeIA()
        def reset(self): pass

    pkg = types.ModuleType("harvesters")
    core = types.ModuleType("harvesters.core")
    core.Harvester = FakeHarvester
    pkg.core = core
    monkeypatch.setitem(sys.modules, "harvesters", pkg)
    monkeypatch.setitem(sys.modules, "harvesters.core", core)

    result = GenTLAdapter().acquire_sequence("GenTL", device_id="SER-HOST", duration_s=0.5,
        frame_rate_hz=4, settings={"cti_file": str(cti), "frame_timeout_s": 0.2})
    assert result["metadata"]["timestamp_source"].startswith("host monotonic")
    assert np.all(np.diff(result["timestamps_s"]) > 0)


def test_camera_parameter_schema_is_connection_specific_for_builtin_transports():
    from thermoprop.integrations.cameras import get_parameter_schema
    # A normal scalar-file workflow should not ask the user for a CTI path.
    assert get_parameter_schema("gentl", "Recorded file") == []
    # A JSON file needs field/unit mapping but not a network URL or credentials.
    file_schema = {x["key"] for x in get_parameter_schema("http_matrix", "Recorded file")}
    assert {"frames_key", "timestamps_key", "units", "calibration_confirmed"} <= file_schema
    assert not {"endpoint_url", "username", "password", "bearer_token"} & file_schema
    # The live HTTP API path exposes endpoint and authentication controls.
    live_schema = {x["key"] for x in get_parameter_schema("http_matrix", "HTTP radiometric matrix API")}
    assert {"endpoint_url", "auth_mode", "password", "bearer_token", "timeout_s"} <= live_schema


def test_example_http_matrix_json_recording_is_loadable():
    from pathlib import Path
    from thermoprop.integrations.cameras import load_vendor_recording
    project_root = Path(__file__).resolve().parents[1]
    payload = (project_root / "examples" / "http_matrix_recording.json").read_bytes()
    result = load_vendor_recording("http_matrix", payload, "http_matrix_recording.json", settings={
        "units": "relative_intensity", "frames_key": "frames", "timestamps_key": "timestamps_s"
    })
    assert result["frames"].shape == (3, 8, 8)
    assert result["metadata"]["units"] == "relative_intensity"
    assert np.allclose(result["timestamps_s"], [0.0, 0.5, 1.0])


def test_export_security_redacts_credentials_in_fields_and_url_queries():
    from thermoprop.security import redact_sensitive_metadata
    source = {
        "password": "secret-pass",
        "bearer_token": "abc123",
        "endpoint_url": "https://camera.local/matrix?stream=thermal&api_key=topsecret&access_token=alsosecret",
        "nested": [{"client_secret": "hidden", "plain": "keep-me"}],
    }
    redacted = redact_sensitive_metadata(source)
    assert redacted["password"] == "[REDACTED]"
    assert redacted["bearer_token"] == "[REDACTED]"
    assert "topsecret" not in redacted["endpoint_url"]
    assert "alsosecret" not in redacted["endpoint_url"]
    assert "stream=thermal" in redacted["endpoint_url"]
    assert redacted["nested"][0]["client_secret"] == "[REDACTED]"
    assert redacted["nested"][0]["plain"] == "keep-me"


def test_gentl_optional_feature_configuration_checks_range_and_reads_back(monkeypatch, tmp_path):
    import sys
    import types
    from types import SimpleNamespace
    from thermoprop.integrations.builtin_camera_adapters import GenTLAdapter

    cti = tmp_path / "mock.cti"
    cti.write_bytes(b"mock")

    class DeviceInfo:
        id_ = "DEV-CONFIG"
        serial_number = "SER-CONFIG"
        vendor = "Example"
        model = "FeatureThermal"

    class Node:
        def __init__(self, value, minimum, maximum, writable=True):
            self.value = value
            self.min = minimum
            self.max = maximum
            self.is_implemented = True
            self.is_available = True
            self.is_writable = writable

    class FakeIA:
        def __init__(self):
            self.remote_device = SimpleNamespace(node_map=SimpleNamespace(
                ExposureTime=Node(1000.0, 100.0, 20000.0),
                Gain=Node(0.0, 0.0, 12.0),
            ))
        def destroy(self): pass

    class FakeHarvester:
        def __init__(self): self.device_info_list = [DeviceInfo()]
        def add_file(self, path, **kwargs): pass
        def update(self): pass
        def create(self, search_key=None): return FakeIA()
        def reset(self): pass

    pkg = types.ModuleType("harvesters")
    core = types.ModuleType("harvesters.core")
    core.Harvester = FakeHarvester
    pkg.core = core
    monkeypatch.setitem(sys.modules, "harvesters", pkg)
    monkeypatch.setitem(sys.modules, "harvesters.core", core)

    result = GenTLAdapter().configure("SER-CONFIG", connection="GenTL", settings={
        "cti_file": str(cti), "apply_exposure_time": True, "exposure_time_us": 2500.0,
        "apply_gain": True, "gain": 3.5,
    })
    assert result["ok"] is True
    assert result["applied"]["exposure_time_us"]["read_back"] == 2500.0
    assert result["applied"]["gain"]["read_back"] == 3.5


def test_gentl_optional_feature_configuration_rejects_out_of_camera_range(monkeypatch, tmp_path):
    import sys
    import types
    from types import SimpleNamespace
    from thermoprop.integrations.builtin_camera_adapters import GenTLAdapter

    cti = tmp_path / "mock.cti"
    cti.write_bytes(b"mock")

    class DeviceInfo:
        serial_number = "SER-RANGE"
        id_ = "DEV-RANGE"

    class Node:
        value = 1000.0
        min = 100.0
        max = 2000.0
        is_implemented = True
        is_available = True
        is_writable = True

    class FakeIA:
        remote_device = SimpleNamespace(node_map=SimpleNamespace(ExposureTime=Node()))
        def destroy(self): pass

    class FakeHarvester:
        device_info_list = [DeviceInfo()]
        def __init__(self): pass
        def add_file(self, path, **kwargs): pass
        def update(self): pass
        def create(self, search_key=None): return FakeIA()
        def reset(self): pass

    pkg = types.ModuleType("harvesters")
    core = types.ModuleType("harvesters.core")
    core.Harvester = FakeHarvester
    pkg.core = core
    monkeypatch.setitem(sys.modules, "harvesters", pkg)
    monkeypatch.setitem(sys.modules, "harvesters.core", core)

    with pytest.raises(ValueError, match="above the camera's"):
        GenTLAdapter().configure("SER-RANGE", connection="GenTL", settings={
            "cti_file": str(cti), "apply_exposure_time": True, "exposure_time_us": 3000.0,
        })
