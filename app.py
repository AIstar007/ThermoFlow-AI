from __future__ import annotations

import hashlib
import io
import json
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from thermoprop.analysis import analyze_sequence
from thermoprop.dashboard import (
    make_arrival_figure, make_frame_player, make_metrics_figure,
    make_named_roi_figure, make_motion_figure,
)
from thermoprop.io import load_image_zip
from thermoprop.preprocess import preprocess_sequence
from thermoprop.protocol import build_protocol_timeline
from thermoprop.quality import compute_quality_report
from thermoprop.recovery import fit_rewarming_time_constant
from thermoprop.roi import (
    validate_regions, rasterize_regions, roi_metrics_from_result, overlap_report,
)
from thermoprop.roi_editor import mount_roi_editor, extract_component_regions
from thermoprop.synthetic import make_propagation_sequence
from thermoprop.visualization import create_summary_figure
from thermoprop.security import redact_sensitive_metadata as _redact_secrets
from thermoprop.integrations.cameras import (
    get_profiles, get_profile, adapter_status, probe_camera, load_vendor_recording,
    CAMERA_FORM_FACTORS, CAMERA_INTERFACES, DETECTOR_TECHNOLOGIES, SPECTRAL_BANDS, DETECTOR_OUTPUT_TYPES, CALIBRATION_STATES,
    get_parameter_schema, acquire_camera_sequence, acquire_multiple_camera_sequences,
    configure_camera, healthcheck_camera,
)
from thermoprop.integrations.signals import (
    parse_reference_csv, compare_reference_signal, signal_template_csv,
)
from thermoprop.integrations.reference_streams import (
    list_lsl_streams, capture_lsl_channel, signal_to_csv_bytes,
)
from thermoprop.integrations.xdf import load_xdf_streams
from thermoprop.integrations.ai_roi import suggest_rois_yolo_seg
from thermoprop.integrations.tracking import log_analysis_run
from thermoprop.modeling import train_grouped_reference_model, feature_table_template, explain_tree_model
from thermoprop.spectral import welch_spectrum, wavelet_scalogram
from thermoprop.feature_engineering import build_participant_feature_table
from thermoprop.pennes import simulate_pennes_slab
from thermoprop.bioheat import wu_style_model_index


# ----------------------------------------------------------------------------
# Page shell and visual system
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Thermal Pattern Propagation Lab v1.5.0", page_icon="🌡️",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--tp-teal:#54E0C0;--tp-blue:#8BA8FF;--tp-bg:#07111f;--tp-muted:#91A3B8;}
html,body,[class*="css"]{font-family:'Manrope',sans-serif;}
.stApp{background:radial-gradient(ellipse at 10% -10%,rgba(84,224,192,.13),transparent 34%),radial-gradient(ellipse at 92% 0%,rgba(139,168,255,.13),transparent 32%),#07111f;color:#e8f0f8;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1727,#091321);border-right:1px solid rgba(148,163,184,.12);}
.block-container{max-width:1600px;padding-top:1.5rem;padding-bottom:3rem;}
.tp-hero{background:linear-gradient(120deg,rgba(18,38,58,.96),rgba(12,25,43,.92));border:1px solid rgba(84,224,192,.22);border-radius:22px;padding:1.35rem 1.55rem;margin-bottom:1rem;box-shadow:0 16px 50px rgba(0,0,0,.18);}
.tp-kicker{color:var(--tp-teal);font-size:.7rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase;}
.tp-title{font-size:clamp(1.8rem,3.2vw,2.65rem);font-weight:800;letter-spacing:-.05em;line-height:1.08;margin:.4rem 0;}
.tp-subtitle{color:#aabbd0;font-size:.92rem;max-width:980px;line-height:1.6;}
.tp-chip{display:inline-flex;border:1px solid rgba(84,224,192,.28);background:rgba(84,224,192,.06);color:#bdf7e9;border-radius:99px;padding:.28rem .62rem;margin:.65rem .35rem 0 0;font-size:.67rem;font-weight:700;letter-spacing:.04em;}
.tp-card{background:rgba(14,29,47,.9);border:1px solid rgba(148,163,184,.13);border-radius:15px;padding:.95rem 1rem;height:100%;}
.tp-label{font-size:.65rem;color:#8ea4ba;text-transform:uppercase;letter-spacing:.1em;font-weight:800;}
.tp-value{font-size:1.15rem;color:#eff8ff;font-weight:800;margin-top:.22rem;}
.tp-note{font-size:.76rem;color:#92a7bd;line-height:1.5;margin-top:.3rem;}
.tp-section{font-size:1.15rem;font-weight:800;letter-spacing:-.025em;margin:.25rem 0 .45rem;}
.tp-mono{font-family:'DM Mono',monospace;}
.tp-live{display:inline-block;width:8px;height:8px;border-radius:50%;background:#54e0c0;margin-right:7px;box-shadow:0 0 0 0 rgba(84,224,192,.55);animation:pulse 1.8s infinite;}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(84,224,192,.5)}70%{box-shadow:0 0 0 9px rgba(84,224,192,0)}100%{box-shadow:0 0 0 0 rgba(84,224,192,0)}}
div[data-testid="stMetric"]{background:rgba(14,29,47,.78);border:1px solid rgba(148,163,184,.12);padding:.75rem .9rem;border-radius:13px;}
div[data-testid="stMetricLabel"]{color:#9eb2c8;}div[data-testid="stMetricValue"]{color:#eef7ff;}
div.stButton>button[kind="primary"]{background:linear-gradient(95deg,#54e0c0,#8ba8ff);color:#071521;border:0;border-radius:11px;font-weight:800;min-height:2.7rem;}
div.stButton>button{border-radius:10px;}[data-testid="stFileUploader"]{border:1px dashed rgba(84,224,192,.4);border-radius:13px;background:rgba(84,224,192,.03);}
.stCaption,small{color:#91a3b8!important;}
hr{border-color:rgba(148,163,184,.13)!important;}
</style>
<div class="tp-hero">
  <div class="tp-kicker">Biomedical thermal dynamics · research workspace</div>
  <div class="tp-title">Thermal Pattern Propagation Lab</div>
  <div class="tp-subtitle">Explore time-resolved surface-temperature patterns, draw anatomical ROIs, inspect acquisition phases, review motion and threshold sensitivity, and export a traceable analysis.</div>
  <div><span class="tp-chip">Interactive frame player</span><span class="tp-chip">Polygon ROI annotation</span><span class="tp-chip">Protocol-aware analysis</span><span class="tp-chip">Research-only · non-diagnostic</span></div>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Thermal cameras observe emitted infrared radiation and may estimate skin-surface temperature when radiometrically calibrated. "
    "This application does not measure blood flow. Direction and speed outputs describe thresholded apparent thermal-pattern propagation only."
)


# ----------------------------------------------------------------------------
# Input helpers
# ----------------------------------------------------------------------------
def _load_uploaded(payload: bytes, suffix: str, npz_key: str | None = None,
                   camera_profile_id: str = "generic", filename: str | None = None,
                   camera_options: dict | None = None) -> dict:
    """Load a scalar thermal sequence or delegate a vendor-native file to its SDK plugin."""
    suffix = suffix.lower()
    native_extensions = {ext for p in get_profiles() if p.id == camera_profile_id for ext in p.native_extensions}
    vendor_metadata = {}
    embedded_timestamps = None
    if suffix in native_extensions:
        recording = load_vendor_recording(camera_profile_id, payload, filename or f"recording{suffix}",
                                         settings=dict(camera_options or {}))
        array = recording["frames"]
        embedded_timestamps = recording.get("timestamps_s")
        vendor_metadata = recording.get("metadata", {}) or {}
    elif suffix == ".npy":
        array = np.load(io.BytesIO(payload), allow_pickle=False)
    elif suffix == ".npz":
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            keys = list(data.files)
            selected = npz_key
            preferred = ("temperature", "temperatures", "temps", "frames", "thermal", "array2")
            if selected is None:
                selected = next((k for k in preferred if k in keys), None)
                if selected is None:
                    candidates = [k for k in keys if np.asarray(data[k]).ndim in (2, 3)]
                    if len(candidates) == 1:
                        selected = candidates[0]
                    else:
                        raise ValueError(f"Choose an image sequence key in the NPZ. Available keys: {keys}")
            if selected not in keys:
                raise ValueError(f"NPZ key {selected!r} is missing. Available keys: {keys}")
            array = data[selected]
            for time_key in ("timestamps_s", "timestamps", "time_s", "times"):
                if time_key in keys and np.asarray(data[time_key]).ndim == 1:
                    embedded_timestamps = np.asarray(data[time_key], dtype=float)
                    break
    elif suffix in {".tif", ".tiff"}:
        import tifffile
        array = tifffile.imread(io.BytesIO(payload))
    elif suffix == ".zip":
        array = load_image_zip(payload)
    elif suffix == ".csv":
        try:
            array = np.loadtxt(io.BytesIO(payload), delimiter=",")
        except ValueError:
            array = pd.read_csv(io.BytesIO(payload)).select_dtypes(include=[np.number]).to_numpy()
    else:
        raise ValueError("Use a scalar array/image export (NPY, NPZ, TIFF, CSV, ZIP) or a supported native vendor recording with its SDK adapter installed.")
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None]
    if array.ndim == 4 and array.shape[-1] in (3, 4):
        raise ValueError("RGB or pseudocolour images are not valid temperature arrays. Use scalar radiometric or grayscale frames.")
    if array.ndim != 3:
        raise ValueError(f"Expected frames × height × width, received {array.shape}.")
    if array.shape[0] < 3 or min(array.shape[1:]) < 8:
        raise ValueError("Need at least 3 frames and image dimensions of at least 8 × 8 pixels.")
    if array.dtype.kind not in "uif":
        raise ValueError(f"Expected numeric thermal values, received {array.dtype}.")
    if not np.isfinite(array).any():
        raise ValueError("No finite pixel values found.")
    if array.nbytes > 1_500_000_000:
        raise ValueError("Decoded sequence is larger than the 1.5 GB safety limit. Split it into smaller segments.")
    return {"frames": array.astype(np.float32, copy=False),
            "timestamps_s": np.asarray(embedded_timestamps, dtype=float) if embedded_timestamps is not None else None,
            "vendor_metadata": vendor_metadata}


def _timestamp_vector(uploaded_file) -> np.ndarray:
    raw = uploaded_file.getvalue()
    # Accept a simple one-column CSV. If a header exists, the fallback parser removes it.
    try:
        arr = np.loadtxt(io.BytesIO(raw), delimiter=",")
    except ValueError:
        df = pd.read_csv(io.BytesIO(raw))
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            raise ValueError("Timestamp CSV needs at least one numeric column in seconds.")
        arr = numeric.iloc[:, 0].to_numpy(dtype=float)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        if arr.shape[1] > 1:
            arr = arr[:, 0]
        else:
            arr = arr[:, 0]
    return arr.reshape(-1)


def _clean(value):
    if isinstance(value, dict): return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_clean(v) for v in value]
    if isinstance(value, np.ndarray): return _clean(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)): return bool(value)
    return value


def _npy_bytes(array) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{str(k): np.asarray(v) for k, v in arrays.items()})
    return buffer.getvalue()


def _figure_png(result, frames, units: str) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "analysis_summary.png"
        create_summary_figure(result, frames, path, unit_label=units)
        return path.read_bytes()


def _export_bundle(result, processed, roi_metrics, regions, timeline, qc, settings, summary_png, player_html,
                   reference_analysis=None, reference_timeseries=None):
    buf = io.BytesIO()
    arrays = io.BytesIO()
    np.savez_compressed(arrays, baseline=result.baseline, delta=result.delta, event_delta=result.event_delta,
                        delta_rate=result.delta_rate,
                        roi_mask=result.roi_mask.astype(np.uint8), warming_first_arrival_s=result.warming_arrival_s,
                        cooling_first_arrival_s=result.cooling_arrival_s, times_s=timeline.times_s)
    summary = {"analysis": result.summary, "protocol": timeline.metadata,
               "phase_spans": timeline.phase_spans, "quality": qc, "settings": settings,
               "regions": regions,
               "regional_recovery_fits": {name: fit for name, fit in settings.get("regional_recovery_fits", {}).items()}}
    manifest = {
        "application": "Thermal Pattern Propagation Lab",
        "application_version": "1.5.0",
        "source_name": settings.get("source_name"),
        "source_sha256": settings.get("source_hash"),
        "timestamps_sha256": settings.get("timestamps_hash"),
        "camera_profile_id": settings.get("camera_profile_id"),
        "camera_model": settings.get("camera_model"),
        "camera_serial": settings.get("camera_serial"),
        "calibration_id": settings.get("calibration_id"),
        "units": settings.get("units"),
        "radiometric_calibration_confirmed": settings.get("calibrated_celsius", False),
        "protocol_mode": settings.get("protocol_mode"),
        "registered": settings.get("register", False),
        "registration_mode": settings.get("registration_mode"),
        "software": {"numpy": np.__version__, "pandas": pd.__version__},
        "interpretation_warning": "Outputs describe skin-surface temperature / relative-intensity patterns. They are not direct blood-flow measurements.",
    }
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(_clean(manifest), indent=2, allow_nan=False))
        z.writestr("summary.json", json.dumps(_clean(summary), indent=2, allow_nan=False))
        z.writestr("union_roi_frame_metrics.csv", result.metrics.to_csv(index=False))
        z.writestr("regional_frame_metrics.csv", pd.concat(roi_metrics.values(), ignore_index=True).to_csv(index=False))
        z.writestr("thermal_analysis_arrays.npz", arrays.getvalue())
        z.writestr("analysis_summary.png", summary_png)
        z.writestr("regions.json", json.dumps(_clean(regions), indent=2))
        z.writestr("protocol_timeline.csv", pd.DataFrame({"frame": np.arange(len(timeline.times_s)),
                                                           "time_s": timeline.times_s,
                                                           "phase": timeline.phase_labels}).to_csv(index=False))
        if player_html:
            z.writestr("interactive_frame_player.html", player_html)
        if reference_analysis:
            z.writestr("reference_signal_analysis.json", json.dumps(_clean(reference_analysis), indent=2, allow_nan=False))
        for name, frame in (reference_timeseries or {}).items():
            z.writestr(f"reference_signals/{name.lower().replace(' ','_')}_aligned.csv", frame.to_csv(index=False))
        z.writestr("README.txt", "Thermal Pattern Propagation Lab export\n\n"
                   "Outputs describe surface-temperature/relative-intensity features. They are not direct blood-flow measurements.\n"
                   "Keep the original camera radiometric files, timestamps and calibration metadata with this export.\n")
    return buf.getvalue(), summary


def _load_demo(name: str):
    if name.startswith("Synthetic warming"):
        return make_propagation_sequence(direction="right", seed=11)
    return make_propagation_sequence(direction="left", seed=12)


def _render_camera_parameter_controls(schema: list[dict], key_prefix: str) -> dict:
    """Render only the parameters exposed by the selected installed camera adapter."""
    values = {}
    for field in schema:
        key = field["key"]; kind = field["type"]; label = field["label"]
        help_text = field.get("help") or None; default = field.get("default", "")
        widget_key = f"{key_prefix}_{key}"
        if kind == "bool":
            values[key] = st.checkbox(label, value=bool(default), help=help_text, key=widget_key)
        elif kind == "select":
            options = list(field["options"])
            index = options.index(str(default)) if str(default) in options else 0
            values[key] = st.selectbox(label, options, index=index, help=help_text, key=widget_key)
        elif kind == "int":
            values[key] = st.number_input(label, min_value=int(field["min"]) if field.get("min") is not None else None,
                max_value=int(field["max"]) if field.get("max") is not None else None, value=int(default),
                step=int(field.get("step", 1)), help=help_text, key=widget_key)
        elif kind == "float":
            values[key] = st.number_input(label, min_value=float(field["min"]) if field.get("min") is not None else None,
                max_value=float(field["max"]) if field.get("max") is not None else None, value=float(default),
                step=float(field.get("step", 0.1)), help=help_text, key=widget_key)
        elif kind == "password":
            values[key] = st.text_input(label, value=str(default), type="password", help=help_text, key=widget_key)
        else:
            values[key] = st.text_input(label, value=str(default), help=help_text, key=widget_key)
    return values


# ----------------------------------------------------------------------------
# Sidebar: sequence and acquisition settings
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌡️ Workspace")
    source = st.radio("Sequence source", [
        "Upload thermal recording", "Live camera acquisition", "Synthetic validation demo"
    ], key="source_mode")
    profiles = get_profiles()
    profile_labels = {p.label: p.id for p in profiles}
    selected_profile_label = st.selectbox("Camera / acquisition profile", list(profile_labels),
        index=list(profile_labels.values()).index("generic"),
        help="Choose the camera actually used. Standard scalar exports are supported independently of vendor SDKs; native files and live acquisition require a matching adapter.")
    camera_profile_id = profile_labels[selected_profile_label]
    selected_camera_profile = get_profile(camera_profile_id)
    camera_profile_status = adapter_status(camera_profile_id)
    connection_options = [c for c in selected_camera_profile.live_connections if c != "Recorded file"]
    camera_connection = st.selectbox("Connection / workflow", ["Recorded file", *connection_options], index=0,
        key=f"camera_connection_{camera_profile_id}",
        help="These are declared interface options, not a guarantee of connection. The selected plugin determines which operations are actually available.")
    camera_parameter_values = {}
    try:
        camera_parameter_schema = get_parameter_schema(camera_profile_id, camera_connection)
    except Exception as exc:
        camera_parameter_schema = []
        st.error(f"Could not load camera-specific parameter schema: {exc}")
    if camera_parameter_schema:
        with st.expander("Camera-specific SDK parameters", expanded=False):
            st.caption("Controls are supplied by the installed adapter and passed to the vendor SDK. Parameters are not assumed to be universal across cameras.")
            camera_parameter_values = _render_camera_parameter_controls(camera_parameter_schema, f"camera_cfg_{camera_profile_id}_{camera_connection}")
    with st.expander("Camera / detector architecture (optional metadata)", expanded=False):
        st.caption("Choose the closest hardware description. These values document the instrument; they do not enable a driver, calibrate pixels, or change vendor SDK settings.")
        arch1, arch2 = st.columns(2)
        camera_form_factor = arch1.selectbox("Camera form factor", CAMERA_FORM_FACTORS,
            key=f"camera_form_factor_{camera_profile_id}")
        detector_architecture = arch2.selectbox("Detector technology", DETECTOR_TECHNOLOGIES,
            key=f"detector_architecture_{camera_profile_id}")
        arch3, arch4 = st.columns(2)
        spectral_band = arch3.selectbox("Spectral band", SPECTRAL_BANDS,
            key=f"spectral_band_{camera_profile_id}")
        detector_output_type = arch4.selectbox("Detector output representation", DETECTOR_OUTPUT_TYPES,
            key=f"detector_output_type_{camera_profile_id}")
        physical_interface = st.selectbox("Physical connection / transport", CAMERA_INTERFACES,
            key=f"camera_physical_interface_{camera_profile_id}",
            help="Descriptive metadata only. Live acquisition still requires a compatible driver/adapter.")
        calibration_state = st.selectbox("Calibration status", CALIBRATION_STATES,
            key=f"camera_calibration_state_{camera_profile_id}")
        custom_vendor_name = ""
        if camera_profile_id == "other":
            custom_vendor_name = st.text_input("Other vendor / OEM name", key="custom_camera_vendor_name",
                placeholder="Enter the manufacturer or OEM module provider")
        pixel_format_label = st.text_input("Pixel format / numeric encoding (optional)",
            placeholder="e.g. Mono16, 14-bit packed, float32 °C", key=f"camera_pixel_format_{camera_profile_id}")
        if detector_output_type.startswith("False-colour"):
            st.warning("False-colour palette images are suitable for visual inspection only. The quantitative pipeline requires scalar per-pixel data; RGB/pseudocolour uploads are rejected.")
        if "temperature" in detector_output_type.casefold() and calibration_state.startswith("Unknown"):
            st.warning("You selected a temperature output but have not documented its calibration status. Verify the export's radiometric meaning before interpreting values as °C/°F.")
    with st.expander("Camera SDK and capability status", expanded=False):
        st.write(f"**{selected_profile_label}**")
        st.caption(camera_profile_status["status_note"])
        st.write("**Offline SDK:** " + camera_profile_status["offline_sdk"])
        st.write("**Live SDK:** " + camera_profile_status["live_sdk"])
        if camera_profile_status["plugin_load_error"]:
            st.error(f"Adapter import error: {camera_profile_status['plugin_load_error']}")
        st.json({k: camera_profile_status[k] for k in [
            "plugin_installed", "adapter_loaded", "native_decode_available", "live_probe_available",
            "live_acquisition_available", "single_camera_acquisition_available",
            "multi_camera_acquisition_available", "healthcheck_available",
            "camera_parameter_schema_available", "configure_available"]}, expanded=False)
        if camera_profile_status.get("adapter_loaded"):
            st.success(f"Adapter loaded: {camera_profile_status['plugin_name']}")
        elif camera_profile_status["plugin_load_error"]:
            st.warning("A camera adapter entry point is installed, but it did not load successfully. See the adapter error above.")
        elif source == "Upload thermal recording" or camera_connection == "Recorded file":
            st.info("No live/native camera adapter is installed for this profile. This does not block standard scalar NPY/NPZ/TIFF/CSV/ZIP imports. Native vendor files and live acquisition need a compatible SDK adapter.")
        else:
            st.warning("No compatible camera adapter is installed. The vendor profile is available for metadata and scalar file imports; install a matching SDK adapter or use the GenICam/GenTL or generic HTTP radiometric-matrix connector when your device supports it.")
        if camera_profile_status["documentation_url"]:
            st.markdown(f"[Vendor / SDK documentation]({camera_profile_status['documentation_url']})")
        adapter_template = Path(__file__).parent / "src" / "thermoprop" / "integrations" / "camera_adapter_template.py.txt"
        if adapter_template.exists():
            st.download_button("Download camera adapter starter template", adapter_template.read_bytes(),
                "camera_adapter_template.py.txt", "text/plain", key="download_camera_adapter_template")
    uploaded = None; frames = None; source_name = ""; source_hash = ""
    vendor_timestamps = None; vendor_metadata = {}; detected_camera = {}
    if source == "Upload thermal recording":
        allowed_extensions = list(selected_camera_profile.standard_extensions)
        if camera_profile_status["native_decode_available"]:
            allowed_extensions += list(selected_camera_profile.native_extensions)
        accepted_types = sorted({ext.lstrip(".") for ext in allowed_extensions})
        uploaded = st.file_uploader("Thermal sequence", type=accepted_types,
            help="Upload scalar radiometric arrays or documented scalar sensor values. Pseudocolour screenshots are not temperature arrays.")
        if uploaded is not None:
            payload = uploaded.getvalue(); source_name = uploaded.name
            source_hash = hashlib.sha256(payload).hexdigest()
            npz_key = None
            if uploaded.name.lower().endswith(".npz"):
                with np.load(io.BytesIO(payload), allow_pickle=False) as z:
                    keys = list(z.files)
                if len(keys) > 1:
                    preferred = next((k for k in ("temperature", "temps", "frames", "array2") if k in keys), keys[0])
                    npz_key = st.selectbox("Thermal array key", keys, index=keys.index(preferred), key=f"npz_key_{source_hash[:10]}")
            try:
                recording = _load_uploaded(payload, Path(uploaded.name).suffix, npz_key, camera_profile_id, uploaded.name,
                                           camera_options=camera_parameter_values)
                frames = recording["frames"]
                vendor_timestamps = recording.get("timestamps_s")
                vendor_metadata = recording.get("vendor_metadata", {})
            except Exception as exc:
                st.error(f"Input error: {exc}")
    elif source == "Live camera acquisition":
        st.caption("Finite capture through the selected SDK adapter. Live acquisition is available only when the installed adapter implements it.")
        if not camera_profile_status["live_acquisition_available"]:
            st.warning("Live acquisition is not available for this camera profile in the current installation. Use recorded-file import or install the matching SDK adapter.")
        elif camera_connection == "Recorded file":
            st.warning("Choose a supported live connection such as USB, Ethernet or GenTL.")
        else:
            discovered_devices = st.session_state.get(f"devices_{camera_profile_id}_{camera_connection}", [])
            if camera_profile_status["live_probe_available"]:
                if st.button("🔎 Discover connected cameras", key=f"probe_{camera_profile_id}_{camera_connection}"):
                    try:
                        discovered_devices = probe_camera(camera_profile_id, camera_connection,
                                                          settings=camera_parameter_values)
                        st.session_state[f"devices_{camera_profile_id}_{camera_connection}"] = discovered_devices
                        st.success(f"Discovered {len(discovered_devices)} camera(s).")
                    except Exception as exc:
                        st.error(f"Camera discovery failed: {exc}")
                discovered_devices = st.session_state.get(f"devices_{camera_profile_id}_{camera_connection}", [])
            device_id = ""
            selected_device_ids = []
            capture_multiple = False
            if discovered_devices:
                device_map = {str(d.get("label", d.get("device_id"))): str(d.get("device_id")) for d in discovered_devices}
                if camera_profile_status["multi_camera_acquisition_available"]:
                    capture_multiple = st.checkbox("Capture multiple cameras in one acquisition",
                        value=not camera_profile_status["single_camera_acquisition_available"],
                        disabled=not camera_profile_status["single_camera_acquisition_available"],
                        key=f"multi_camera_{camera_profile_id}_{camera_connection}",
                        help="Uses multiple device IDs handled by this one installed vendor adapter. Recordings remain separate; heterogeneous vendor cameras need a custom orchestration adapter.")
                if capture_multiple:
                    chosen_labels = st.multiselect("Cameras to capture", list(device_map),
                        default=list(device_map)[:min(2, len(device_map))], key=f"multi_devices_{camera_profile_id}_{camera_connection}")
                    selected_device_ids = [device_map[label] for label in chosen_labels]
                    if len(selected_device_ids) < 2:
                        extra_ids = st.text_input("Additional device IDs (comma-separated)",
                            key=f"extra_multi_devices_{camera_profile_id}_{camera_connection}",
                            help="Use this only for additional devices understood by the adapter.")
                        selected_device_ids.extend(v.strip() for v in extra_ids.split(",") if v.strip())
                        selected_device_ids = list(dict.fromkeys(selected_device_ids))
                    device_id = selected_device_ids[0] if selected_device_ids else ""
                    detected_camera = next((d for d in discovered_devices if str(d.get("device_id")) == device_id), {}) if device_id else {}
                else:
                    device_label = st.selectbox("Connected device", list(device_map), key=f"device_{camera_profile_id}_{camera_connection}")
                    device_id = device_map[device_label]
                    selected_device_ids = [device_id]
                    detected_camera = next((d for d in discovered_devices if str(d.get("device_id")) == device_id), {})
            else:
                if camera_profile_status["multi_camera_acquisition_available"] and not camera_profile_status["single_camera_acquisition_available"]:
                    manual_ids = st.text_input("Camera device IDs (comma-separated)",
                        key=f"manual_devices_{camera_profile_id}_{camera_connection}",
                        help="Enter at least two unique device IDs as understood by the installed adapter.")
                    selected_device_ids = [v.strip() for v in manual_ids.split(",") if v.strip()]
                    capture_multiple = True
                    device_id = selected_device_ids[0] if selected_device_ids else ""
                else:
                    device_id = st.text_input("Device identifier / IP (if required by adapter)", key=f"manual_device_{camera_profile_id}_{camera_connection}")
                    selected_device_ids = [device_id] if device_id else []
            if camera_profile_status["healthcheck_available"] and st.button("Run non-destructive camera health check", key=f"healthcheck_{camera_profile_id}_{camera_connection}"):
                try:
                    health = healthcheck_camera(camera_profile_id, device_id=device_id or None,
                                                connection=camera_connection, settings=camera_parameter_values)
                    st.success("Camera health check passed." if health.get("ok", True) else "Camera health check reported a problem.")
                    st.json(health, expanded=False)
                except Exception as exc:
                    st.error(f"Camera health check failed: {exc}")
            ccap1, ccap2 = st.columns(2)
            capture_duration = ccap1.number_input("Capture duration (seconds)", min_value=0.5, max_value=600.0,
                                                   value=30.0, step=5.0, key=f"capture_duration_{camera_profile_id}")
            capture_fps = ccap2.number_input("Requested frame rate (Hz)", min_value=0.1, max_value=240.0,
                                             value=3.0, step=1.0, key=f"capture_fps_{camera_profile_id}")
            if st.button("● Capture thermal sequence", type="primary", width="stretch", key=f"capture_{camera_profile_id}_{camera_connection}"):
                try:
                    with st.spinner("Configuring and capturing through the selected vendor SDK…"):
                        configuration_result = {}
                        if camera_profile_status["configure_available"]:
                            if capture_multiple:
                                configuration_result = {selected_id: configure_camera(
                                    camera_profile_id, device_id=selected_id,
                                    settings=camera_parameter_values, connection=camera_connection)
                                    for selected_id in selected_device_ids}
                            else:
                                configuration_result = configure_camera(
                                    camera_profile_id, device_id=device_id or None,
                                    settings=camera_parameter_values, connection=camera_connection)
                        if capture_multiple:
                            if len(selected_device_ids) < 2:
                                raise ValueError("Select or enter at least two device IDs for multi-camera capture.")
                            captured = acquire_multiple_camera_sequences(
                                camera_profile_id, camera_connection, selected_device_ids,
                                duration_s=float(capture_duration), frame_rate_hz=float(capture_fps),
                                settings=camera_parameter_values)
                            captured["metadata"].update({"connection": camera_connection,
                                "camera_parameter_values": _redact_secrets(camera_parameter_values),
                                "configuration_result": _redact_secrets(configuration_result),
                                "requested_duration_s": float(capture_duration),
                                "requested_frame_rate_hz": float(capture_fps)})
                            st.session_state[f"live_recording_{camera_profile_id}"] = {"kind": "multi", **captured}
                        else:
                            recording = acquire_camera_sequence(camera_profile_id, camera_connection,
                                device_id=device_id or None, duration_s=float(capture_duration),
                                frame_rate_hz=float(capture_fps), settings=camera_parameter_values)
                            recording["metadata"].update({"device_id": device_id or None, "connection": camera_connection,
                                "camera_parameter_values": _redact_secrets(camera_parameter_values),
                                "configuration_result": _redact_secrets(configuration_result),
                                "requested_duration_s": float(capture_duration), "requested_frame_rate_hz": float(capture_fps)})
                            st.session_state[f"live_recording_{camera_profile_id}"] = recording
                    st.success("Multi-camera capture completed." if capture_multiple else f"Captured {len(recording['frames']):,} frames.")
                except Exception as exc:
                    st.error(f"Camera capture failed: {exc}")
            saved_capture = st.session_state.get(f"live_recording_{camera_profile_id}")
            if saved_capture:
                if saved_capture.get("kind") == "multi":
                    multi_recordings = saved_capture["recordings"]
                    labels = {f"{device_id} · {len(rec['frames']):,} frames": device_id for device_id, rec in multi_recordings.items()}
                    selected_multi_label = st.selectbox("Select camera sequence to analyse", list(labels), key=f"multi_analyse_{camera_profile_id}")
                    recording = multi_recordings[labels[selected_multi_label]]
                    vendor_metadata = {**saved_capture.get("metadata", {}), **recording.get("metadata", {}),
                                       "multi_camera_device_ids": saved_capture.get("device_ids", []),
                                       "selected_device_id": labels[selected_multi_label]}
                else:
                    recording = saved_capture
                    vendor_metadata = recording.get("metadata", {})
                frames = recording["frames"]
                vendor_timestamps = recording.get("timestamps_s")
                source_name = f"Live capture · {selected_profile_label}"
                source_hash = hashlib.sha256(np.asarray(frames).tobytes()).hexdigest()
                st.success(f"Captured sequence available · {len(frames):,} frames")
                if st.button("Discard captured sequence", key=f"discard_capture_{camera_profile_id}"):
                    st.session_state.pop(f"live_recording_{camera_profile_id}", None)
                    st.rerun()
    else:
        demo_name = st.selectbox("Choose a known-motion demo", ["Synthetic warming front →", "Synthetic cooling front ←"])
        source_name = demo_name; source_hash = f"demo::{demo_name}"
        frames, demo_truth = _load_demo(demo_name)
        st.info("Deterministic synthetic sequence. Use it to validate software behaviour, not biological performance.")

    if frames is not None:
        n, h, w = frames.shape
        st.success(f"Loaded {n:,} frames · {w} × {h} px")
        st.caption(f"Memory footprint: {frames.nbytes / 1024**2:.1f} MiB")
        if n > 900:
            st.warning("Long recording: browser playback will sample frames, but numerical analysis uses the full sequence.")

        st.divider(); st.markdown("### Acquisition")
        unit_options = ["°C", "°F", "radiance", "relative intensity", "camera counts"]
        metadata_units_raw = str(vendor_metadata.get("units", "")).strip()
        metadata_unit_aliases = {
            "camera_counts": "camera counts", "raw_counts": "camera counts", "counts": "camera counts",
            "relative_intensity": "relative intensity", "normalized_intensity": "relative intensity",
            "celsius": "°C", "degc": "°C", "fahrenheit": "°F", "degf": "°F",
        }
        metadata_units = metadata_unit_aliases.get(metadata_units_raw.casefold(), metadata_units_raw)
        output_unit_hints = {
            "Calibrated radiometric temperature (°C)": "°C",
            "Calibrated radiometric temperature (°F)": "°F",
            "Radiance / calibrated signal": "radiance",
            "Linearized detector counts": "camera counts",
            "Raw detector counts / digital levels": "camera counts",
            "Relative / normalized intensity": "relative intensity",
        }
        if source.startswith("Synthetic"):
            default_units = "°C"
        elif metadata_units in unit_options:
            default_units = metadata_units
        else:
            default_units = output_unit_hints.get(detector_output_type, "relative intensity")
        units = st.selectbox("Pixel-value units", unit_options,
                             index=unit_options.index(default_units), key=f"units_{source_hash[:10]}",
                             help="This labels values; it does not calibrate the source. Select °C only for radiometrically calibrated temperature arrays; radiance/counts are not surface temperature.")
        calibrated = st.checkbox("I confirm radiometric calibration in °C", value=False, disabled=units != "°C", key=f"calibrated_{source_hash[:10]}")
        fps_default = 2.0 if source.startswith("Synthetic") else 3.0
        fps = st.number_input("Nominal frame rate (Hz)", min_value=0.01, max_value=240.0, value=fps_default, step=0.5, key=f"fps_{source_hash[:10]}")
        timestamps_file = st.file_uploader("Optional frame timestamp CSV (seconds)", type=["csv"], key=f"timestamps_{source_hash[:10]}")
        timestamp_values = np.asarray(vendor_timestamps, dtype=float) if vendor_timestamps is not None else None
        if timestamp_values is not None:
            if len(timestamp_values) == n:
                st.success("Using timestamps embedded by the camera SDK")
            else:
                st.warning("Vendor-provided timestamp count does not match the frame count; ignoring embedded timestamps.")
                timestamp_values = None
        if timestamps_file is not None:
            try:
                timestamp_values = _timestamp_vector(timestamps_file)
                if len(timestamp_values) != n:
                    st.error(f"Timestamp count {len(timestamp_values)} does not match frame count {n}.")
                    timestamp_values = None
                else:
                    st.success(f"Loaded {len(timestamp_values)} timestamps")
            except Exception as exc:
                st.error(f"Timestamp file error: {exc}")

        st.divider(); st.markdown("### Protocol timeline")
        protocol_mode = st.radio("Recording protocol", [
            "Baseline + rewarming (challenge interval not recorded)",
            "Continuous baseline → challenge → recovery",
        ], help="Choose the mode that matches the actual recording. Do not create a challenge segment if it was not recorded.")
        mode_id = "baseline_recovery_unrecorded_challenge" if protocol_mode.startswith("Baseline +") else "continuous_challenge_recovery"
        baseline_default = min(90 if n > 180 and not source.startswith("Synthetic") else 4 if source.startswith("Synthetic") else 10, n - 1)
        baseline_frames = st.number_input("Baseline frames", min_value=2, max_value=n-1,
                                          value=max(2, int(baseline_default)), step=1, key=f"baseline_frames_{source_hash[:10]}")
        challenge_start = challenge_end = None
        gap_known = False; gap_seconds = None; timestamps_include_gap = False
        if mode_id == "baseline_recovery_unrecorded_challenge":
            if timestamp_values is None:
                gap_known = st.checkbox("I have independently confirmed the unrecorded challenge duration", value=False)
                if gap_known:
                    gap_seconds = st.number_input("Unrecorded challenge duration (seconds)", min_value=0.0,
                                                   value=60.0, step=1.0)
            else:
                timestamps_include_gap = st.checkbox("Timestamps include the documented unrecorded challenge interval", value=False,
                    help="Only check this when the timestamp clock spans the missing cooling/challenge interval and that interval is documented.")
        else:
            challenge_start = st.number_input("Challenge start frame", min_value=int(baseline_frames), max_value=n-2,
                                              value=min(int(baseline_frames), n-2), step=1)
            challenge_end = st.number_input("Challenge end frame (exclusive)", min_value=int(challenge_start)+1,
                                            max_value=n-1, value=min(int(challenge_start)+max(2, int(fps*10)), n-1), step=1)

        front_reference_frame = int(baseline_frames) if mode_id == "baseline_recovery_unrecorded_challenge" else int(challenge_start)
        event_reference_max = max(1, min(10, n - front_reference_frame))
        event_reference_frames = st.number_input("Event reference window (frames)", min_value=1,
            max_value=event_reference_max, value=1,
            help="Median reference at the first recorded rewarming/challenge frames. A short window reduces noise; a longer window may blur a fast-moving front.")

        st.divider(); st.markdown("### Signal processing")
        threshold = st.number_input("Minimum Δ threshold", min_value=0.0,
                                    value=0.3 if source.startswith("Synthetic") else 0.25, step=0.05,
                                    help="Same units as the image values. An adaptive robust baseline-noise floor is applied as well.")
        spatial_sigma = st.slider("Spatial Gaussian σ (pixels)", 0.0, 3.0, 0.5, 0.1)
        temporal_sigma = st.slider("Temporal smoothing σ (frames)", 0.0, 3.0, 0.0, 0.1)
        register = st.toggle("Enable motion registration", value=False,
                             help="Registration can suppress genuine moving thermal patterns. Inspect every transform before interpreting propagation.")
        registration_mode_label = st.selectbox("Registration model", ["Translation (phase correlation)", "Affine (ECC / OpenCV)"],
            help="Affine registration can correct small rotation/scale/shear; large or deformable motion still needs anatomical landmark tracking.",
            disabled=not register)
        registration_mode = "phase_translation" if registration_mode_label.startswith("Translation") else "affine_ecc"
        physical_scale_known = st.toggle("Use documented spatial calibration", value=False)
        pixel_size_mm = st.number_input("Pixel pitch (mm/pixel)", min_value=0.0001, value=1.0, step=0.01,
                                         format="%.4f", disabled=not physical_scale_known)

        st.divider(); st.markdown("### Acquisition context")
        ambient_c = st.number_input("Ambient temperature (°C)", value=21.0, step=0.5)
        humidity = st.number_input("Relative humidity (%)", min_value=0.0, max_value=100.0, value=45.0, step=1.0)
        acclimatization = st.number_input("Acclimatization (minutes)", min_value=0.0, value=15.0, step=1.0)
        camera_distance = st.number_input("Camera distance (cm)", min_value=0.0, value=100.0, step=5.0)
        emissivity = st.number_input("Emissivity setting", min_value=0.01, max_value=1.0, value=0.98, step=0.01)
        reflected_temperature = st.number_input("Reflected apparent temperature (°C)", value=20.0, step=0.5)
        transmissivity = st.number_input("Atmospheric / window transmissivity", min_value=0.01, max_value=1.0, value=1.0, step=0.01)
        window_temperature = st.number_input("Window temperature (°C; if applicable)", value=20.0, step=0.5)
        camera_model = st.text_input("Camera model", value=str(detected_camera.get("model", "") or ""), placeholder="e.g. FLIR A655sc / Seek Mosaic / Optris PI 640", key=f"camera_model_{camera_profile_id}")
        camera_serial = st.text_input("Camera serial / identifier (optional)", value=str(detected_camera.get("serial", detected_camera.get("device_id", "")) or ""), key=f"camera_serial_{camera_profile_id}")
        calibration_id = st.text_input("Calibration certificate / profile ID (optional)", key=f"calibration_id_{camera_profile_id}")
        notes = st.text_area("Protocol / acquisition notes", height=75, placeholder="Cooling method, camera angle, room stability, sensor settings…")
        st.caption("Radiometric settings above are stored as acquisition metadata. They are not reapplied to already-decoded temperature arrays; only a validated vendor SDK decoder should perform camera-specific radiometric correction.")

        st.divider(); st.markdown("### Optional physiological reference signals")
        st.caption("Add synchronized CSV channels, LabRecorder XDF streams, or capture a finite channel from an LSL outlet. A channel is only a reference measurement when the sensor itself is appropriate and its clock alignment is documented.")
        sensor_labels = ["LSCI perfusion index", "Laser-Doppler flux", "PPG amplitude / envelope", "ECG", "GSR / EDA", "External probe temperature", "Ambient temperature logger", "Other reference signal"]
        sensor_modalities = {
            "LSCI perfusion index": "lsci", "Laser-Doppler flux": "laser-doppler",
            "PPG amplitude / envelope": "ppg", "ECG": "ecg", "GSR / EDA": "eda",
            "External probe temperature": "probe-temperature", "Ambient temperature logger": "ambient-temperature",
            "Other reference signal": "other",
        }
        reference_imports_key = f"tp_reference_imports_{source_hash[:12]}"
        sensor_uploads = {}
        for saved in st.session_state.get(reference_imports_key, {}).values():
            sensor_uploads[saved["name"]] = {k: v for k, v in saved.items() if k != "name"}

        selected_sensors = st.multiselect("Reference channels to include", sensor_labels,
            help="Choose modalities you actually recorded. Multiple channels can be combined, each with its own units and clock offset.")
        for sensor_name in selected_sensors:
            with st.expander(f"CSV: {sensor_name}", expanded=False):
                sf = st.file_uploader(f"{sensor_name} CSV", type=["csv"], key=f"sensor_file_{source_hash[:8]}_{sensor_name}")
                if sf is not None:
                    sensor_bytes = sf.getvalue()
                    try:
                        sdf = pd.read_csv(io.BytesIO(sensor_bytes))
                        numeric_cols = [c for c in sdf.columns if pd.to_numeric(sdf[c], errors="coerce").notna().sum() >= 5]
                        if len(numeric_cols) >= 2:
                            default_time = next((c for c in numeric_cols if any(k in c.casefold() for k in ["time", "timestamp", "sec"])), numeric_cols[0])
                            time_col = st.selectbox("Timestamp column (seconds)", numeric_cols, index=numeric_cols.index(default_time), key=f"sensor_time_{source_hash[:8]}_{sensor_name}")
                            value_cols = [c for c in numeric_cols if c != time_col]
                            value_col = st.selectbox("Signal value column", value_cols, key=f"sensor_value_{source_hash[:8]}_{sensor_name}")
                            offset = st.number_input("Clock offset to add (seconds)", value=0.0, step=0.1, key=f"sensor_offset_{source_hash[:8]}_{sensor_name}")
                            sig_units = st.text_input("Signal units", value="a.u.", key=f"sensor_units_{source_hash[:8]}_{sensor_name}")
                            processing = st.selectbox("Signal preprocessing", ["Raw", "Light smoothing"], key=f"sensor_processing_{source_hash[:8]}_{sensor_name}")
                            sensor_uploads[sensor_name] = {"payload": sensor_bytes, "filename": sf.name,
                                "time_column": time_col, "value_column": value_col, "time_offset_s": float(offset),
                                "units": sig_units, "modality": sensor_modalities[sensor_name],
                                "processing": "smooth" if processing == "Light smoothing" else "raw",
                                "hash": hashlib.sha256(sensor_bytes).hexdigest(), "origin": "CSV upload"}
                        else:
                            st.error("CSV needs numeric timestamp and value columns with at least five finite rows.")
                    except Exception as exc:
                        st.error(f"Could not inspect reference CSV: {exc}")

        with st.expander("Import synchronized LabRecorder XDF", expanded=False):
            st.caption("XDF can preserve multi-stream timestamps and clock-correction metadata. pyxdf support is optional; retain the original XDF file as the authoritative acquisition record.")
            xdf_file = st.file_uploader("LabRecorder XDF file", type=["xdf"], key=f"xdf_file_{source_hash[:8]}")
            xdf_sync_clocks = st.checkbox("Apply XDF clock synchronization", value=True, key=f"xdf_sync_{source_hash[:8]}",
                help="Use the recorded LSL clock-offset history to map streams to the recording clock domain.")
            xdf_dejitter = st.checkbox("De-jitter regular-rate timestamps", value=False, key=f"xdf_dejitter_{source_hash[:8]}",
                help="Off by default to preserve observed irregular intervals and dropped-frame gaps. Enable only for streams known to be regularly sampled.")
            if xdf_file is not None:
                xdf_payload = xdf_file.getvalue()
                xdf_hash = hashlib.sha256(xdf_payload).hexdigest()
                xdf_config_key = hashlib.sha256(f"{xdf_hash}:{int(xdf_sync_clocks)}:{int(xdf_dejitter)}".encode()).hexdigest()
                xdf_cache = st.session_state.setdefault("tp_xdf_cache", {})
                if st.button("Read XDF stream metadata", key=f"read_xdf_{xdf_config_key[:10]}"):
                    try:
                        with st.spinner("Reading XDF streams and clock metadata…"):
                            xdf_cache[xdf_config_key] = load_xdf_streams(xdf_payload, xdf_file.name,
                                synchronize_clocks=bool(xdf_sync_clocks), dejitter_timestamps=bool(xdf_dejitter))
                        st.session_state["tp_xdf_cache"] = xdf_cache
                        st.success(f"Loaded {len(xdf_cache[xdf_config_key])} numeric streams.")
                    except Exception as exc:
                        st.error(f"XDF import unavailable or failed: {exc}")
                xdf_streams = st.session_state.get("tp_xdf_cache", {}).get(xdf_config_key, [])
                if xdf_streams:
                    stream_labels = [f"{i}: {x['name']} · {x['type'] or 'no type'} · {x['sample_count']} samples" for i, x in enumerate(xdf_streams)]
                    stream_idx = st.selectbox("XDF stream", range(len(stream_labels)), format_func=lambda i: stream_labels[i], key=f"xdf_stream_{xdf_hash[:10]}")
                    xdf_stream = xdf_streams[int(stream_idx)]
                    channel_idx = st.selectbox("Channel", range(xdf_stream["channel_count"]),
                        format_func=lambda i: f"{i}: {xdf_stream['channel_labels'][i]}", key=f"xdf_channel_{xdf_hash[:10]}_{stream_idx}")
                    xdf_modality_label = st.selectbox("Signal modality", sensor_labels, key=f"xdf_modality_{xdf_hash[:10]}_{stream_idx}_{channel_idx}")
                    xdf_units = st.text_input("Signal units", value="a.u.", key=f"xdf_units_{xdf_hash[:10]}_{stream_idx}_{channel_idx}")
                    xdf_offset = st.number_input("Clock offset to add to XDF timestamps (seconds)", value=0.0, step=0.001, format="%.6f", key=f"xdf_offset_{xdf_hash[:10]}_{stream_idx}_{channel_idx}")
                    xdf_smooth = st.checkbox("Apply light smoothing after import", value=False, key=f"xdf_smooth_{xdf_hash[:10]}_{stream_idx}_{channel_idx}")
                    if st.button("Add selected XDF channel", key=f"add_xdf_{xdf_hash[:10]}_{stream_idx}_{channel_idx}"):
                        try:
                            stream_data = xdf_streams[int(stream_idx)]
                            values = np.asarray(stream_data["samples"][:, int(channel_idx)], dtype=float)
                            stream_times = np.asarray(stream_data["timestamps_s"], dtype=float)
                            finite_channel = np.isfinite(values) & np.isfinite(stream_times)
                            if int(finite_channel.sum()) < 5:
                                raise ValueError("Selected XDF channel has fewer than five finite timestamped samples.")
                            payload_csv = signal_to_csv_bytes(stream_times[finite_channel], values[finite_channel])
                            label = f"XDF {xdf_stream['name']} · {xdf_stream['channel_labels'][int(channel_idx)]}"
                            key = f"xdf:{xdf_hash}:{stream_idx}:{channel_idx}"
                            imports = st.session_state.setdefault(reference_imports_key, {})
                            imports[key] = {"name": label, "payload": payload_csv, "filename": xdf_file.name,
                                "time_column": "time_s", "value_column": "value", "time_offset_s": float(xdf_offset),
                                "units": xdf_units, "modality": sensor_modalities[xdf_modality_label],
                                "processing": "smooth" if xdf_smooth else "raw", "origin": "XDF import",
                                "xdf_stream_metadata": {**{k: v for k, v in stream_data.items() if k not in {"timestamps_s", "samples"}},
                                                        "selected_channel_nonfinite_samples_removed": int((~finite_channel).sum())}}
                            st.session_state[reference_imports_key] = imports
                            st.success(f"Added {label} to this analysis.")
                        except Exception as exc:
                            st.error(f"Could not add XDF channel: {exc}")
                else:
                    st.info("Upload an XDF file and select Read XDF stream metadata. Install the sensors extra if pyxdf is unavailable.")
            else:
                st.info("Install optional support with: pip install -e '.[sensors]'")

        with st.expander("Discover and capture Lab Streaming Layer (LSL)", expanded=False):
            st.caption("Discover compatible LSL outlets and capture one numeric channel over a finite interval. LSL clock correction maps the outlet timestamps into this machine's LSL clock; if the thermal camera uses another clock, measure and enter the camera-to-LSL offset below.")
            if st.button("Discover LSL streams", key=f"discover_lsl_{source_hash[:8]}"):
                try:
                    with st.spinner("Resolving LSL streams on the local network…"):
                        streams = list_lsl_streams(wait_time_s=1.5)
                    st.session_state["tp_lsl_streams"] = streams
                    st.success(f"Found {len(streams)} LSL stream(s).")
                except Exception as exc:
                    st.error(f"LSL discovery unavailable or failed: {exc}")
            lsl_streams = st.session_state.get("tp_lsl_streams", [])
            if lsl_streams:
                lsl_labels = [f"{i}: {x['name']} · {x['type'] or 'no type'} · {x['hostname'] or 'host'} · {x['nominal_srate'] or 'irregular'} Hz · {(x.get('uid') or x.get('source_id') or 'no ID')[:12]}" for i, x in enumerate(lsl_streams)]
                lsl_idx = st.selectbox("LSL stream", range(len(lsl_labels)), format_func=lambda i: lsl_labels[i], key=f"lsl_stream_{source_hash[:8]}")
                lsl_stream = lsl_streams[int(lsl_idx)]
                lsl_channel = st.selectbox("Channel index", range(lsl_stream["channel_count"]), key=f"lsl_channel_{source_hash[:8]}_{lsl_idx}")
                lsl_modality_label = st.selectbox("Signal modality", sensor_labels, key=f"lsl_modality_{source_hash[:8]}_{lsl_idx}_{lsl_channel}")
                lsl_units = st.text_input("Signal units", value="a.u.", key=f"lsl_units_{source_hash[:8]}_{lsl_idx}_{lsl_channel}")
                lsl_clock_correct = st.checkbox("Apply LSL inlet clock correction", value=True,
                    key=f"lsl_clock_correct_{source_hash[:8]}_{lsl_idx}",
                    help="Corrects LSL outlet timestamps into the receiver LSL clock domain. It does not synchronize an independent camera clock.")
                lsl_offset = st.number_input("Clock offset to add (LSL clock → thermal clock, seconds)", value=0.0, step=0.001, format="%.6f", key=f"lsl_offset_{source_hash[:8]}_{lsl_idx}_{lsl_channel}")
                lsl_duration = st.number_input("Capture duration (seconds)", min_value=0.5, max_value=3600.0, value=30.0, step=5.0, key=f"lsl_duration_{source_hash[:8]}_{lsl_idx}")
                if st.button("Capture selected LSL channel", key=f"capture_lsl_{source_hash[:8]}_{lsl_idx}_{lsl_channel}"):
                    try:
                        with st.spinner("Capturing timestamped physiological samples…"):
                            stream_uid = str(lsl_stream.get("uid") or "").strip()
                            source_id = str(lsl_stream.get("source_id") or "").strip()
                            captured = capture_lsl_channel(source_id or None, stream_uid=stream_uid or None,
                                channel_index=int(lsl_channel), duration_s=float(lsl_duration),
                                clock_correct=bool(lsl_clock_correct))
                        payload_csv = signal_to_csv_bytes(captured["timestamps_s"], captured["values"])
                        label = f"LSL {lsl_stream['name']} · channel {int(lsl_channel)}"
                        key = f"lsl:{lsl_stream.get('uid') or lsl_stream.get('source_id')}:{int(lsl_channel)}"
                        imports = st.session_state.setdefault(reference_imports_key, {})
                        imports[key] = {"name": label, "payload": payload_csv, "filename": f"{label}.csv",
                            "time_column": "time_s", "value_column": "value", "time_offset_s": float(lsl_offset),
                            "units": lsl_units, "modality": sensor_modalities[lsl_modality_label],
                            "processing": "raw", "origin": "LSL capture", "lsl_metadata": captured["metadata"]}
                        st.session_state[reference_imports_key] = imports
                        st.success(f"Captured {len(captured['values']):,} samples from {label}.")
                    except Exception as exc:
                        st.error(f"LSL capture failed: {exc}")
            else:
                st.info("LSL is optional. Install it with pip install -e '.[sensors]', then discover active outlets.")

        st.subheader("Reference channels in this analysis")
        for key, saved in list(st.session_state.get(reference_imports_key, {}).items()):
            sensor_uploads[saved["name"]] = {k: v for k, v in saved.items() if k != "name"}
            st.caption(f"{saved['name']} · {saved.get('origin', 'import')} · {saved.get('units', 'a.u.')}")
            if st.button(f"Remove {saved['name']}", key=f"remove_sensor_{hashlib.sha256(key.encode()).hexdigest()[:10]}"):
                imports = st.session_state.get(reference_imports_key, {})
                imports.pop(key, None)
                st.session_state[reference_imports_key] = imports
                st.rerun()
        st.caption("LSCI / laser-Doppler can provide independent perfusion-reference signals when properly acquired. PPG, ECG and EDA are complementary physiology channels, not direct local perfusion ground truth.")
        st.divider()
        run = st.button("▶  Run full analysis", type="primary", width="stretch")
    else:
        run = False
        st.caption("Load a thermal sequence to configure the analysis.")


# ----------------------------------------------------------------------------
# Empty state
# ----------------------------------------------------------------------------
if frames is None:
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("<div class='tp-section'>Start with a thermal sequence</div>", unsafe_allow_html=True)
        st.write("Upload one radiometric or scalar thermal sequence, or use a known-motion demo to check the visualization and analysis workflow.")
        st.code("frames × height × width", language="text")
        st.info("For DIRT-SR38, combine one participant's 90-frame baseline and 720-frame rewarming images from one modality, in temporal order. Do not mix HR and LR folders or different participants.")
    with right:
        st.markdown("<div class='tp-section'>Research safeguards</div>", unsafe_allow_html=True)
        for item in ["Do not interpret normalized PNG values as °C", "Keep missing challenge intervals explicit", "Record acquisition timestamps and calibration", "Use independent perfusion references for validation"]:
            st.markdown(f"✓ {item}")
    st.stop()

n, h, w = frames.shape
st.markdown(f"<div style='display:flex;gap:1rem;flex-wrap:wrap;color:#9eb0c3;font-size:.82rem;margin:.25rem 0 1rem'><span><b style='color:#eef7ff'>{source_name}</b></span><span>{n:,} frames</span><span>{w} × {h} px</span><span>{n/fps:.1f} s nominal record time</span><span>{units}</span></div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Protocol timeline and ROI editor
# ----------------------------------------------------------------------------
try:
    timeline = build_protocol_timeline(
        n_frames=n, fps=float(fps), baseline_frames=int(baseline_frames), mode=mode_id,
        timestamps_s=timestamp_values, challenge_start_frame=int(challenge_start) if challenge_start is not None else None,
        challenge_end_frame=int(challenge_end) if challenge_end is not None else None,
        challenge_gap_seconds=float(gap_seconds) if gap_seconds is not None else None,
        gap_duration_confirmed=bool(gap_known), timestamps_include_gap=bool(timestamps_include_gap),
    )
except Exception as exc:
    st.error(f"Protocol configuration error: {exc}")
    st.stop()

with st.expander("01 · Annotate anatomical regions of interest", expanded=True):
    st.markdown("<div class='tp-section'>Draw one or more named ROIs</div>", unsafe_allow_html=True)
    st.caption("Use Polygon for anatomical outlines or Rectangle for a regular region. Name each region before drawing and click Finish shape to save it. Coordinates are retained in source pixels.")
    seq_key = source_hash[:12]
    region_session_key = f"tp_regions_{seq_key}"
    initial_regions = st.session_state.get(region_session_key, [])
    editor_frame_idx = st.slider("ROI reference frame", 0, n-1, min(int(baseline_frames), n-1),
                                 help="Choose the clearest frame for drawing anatomical boundaries. ROI coordinates remain in source pixels.")
    component_result = mount_roi_editor(st, frames[editor_frame_idx], initial_regions,
                                        key=f"tp_roi_editor_{seq_key}")
    try:
        regions = validate_regions(extract_component_regions(component_result, initial_regions), (h,w))
        st.session_state[region_session_key] = regions
    except Exception as exc:
        st.error(f"ROI geometry issue: {exc}")
        regions = validate_regions(initial_regions, (h,w))
    if regions:
        analysis_regions = regions
        masks = rasterize_regions(regions, (h,w))
        overlap = overlap_report(masks)
        union_mask = np.logical_or.reduce(list(masks.values()))
        roi_cols = st.columns(min(4, len(regions)))
        for col, region in zip(roi_cols, regions[:4]):
            col.metric(region["name"], f"{int(masks[region['id']].sum()):,} px")
        if overlap["overlap_pixels"]:
            st.info(f"Named ROIs overlap over {overlap['overlap_pixels']:,} pixels ({overlap['overlap_fraction_of_union']:.1%} of union). Individual results remain separate.")
    else:
        # Use a clearly named full-frame analysis region without persisting it to
        # the drawing editor. This keeps regional plots and export bundles valid
        # even when the user chooses not to draw anatomical ROIs.
        full_frame = {"id": "full_frame", "name": "Full frame", "color": "#CBD5E1",
                      "points": [[0.0, 0.0], [float(w-1), 0.0],
                                 [float(w-1), float(h-1)], [0.0, float(h-1)]]}
        analysis_regions = [full_frame]
        masks = {"full_frame": np.ones((h,w), dtype=bool)}
        union_mask = masks["full_frame"]
        overlap = {"overlap_pixels":0,"overlap_fraction_of_union":0.0}
        st.info("No named anatomical ROIs: the analysis will use a separate Full frame region. It will not be saved into your ROI editor.")

with st.expander("02 · AI-assisted anatomical ROI suggestions (optional)", expanded=False):
    st.warning("Use only a trusted YOLO segmentation model trained or validated on thermal imagery from a comparable camera/domain. Generic RGB models are not assumed to work on thermal frames. AI masks are suggestions and must be reviewed before quantitative analysis.")
    ai_weights = st.file_uploader("Upload trusted YOLO segmentation weights (.pt / .onnx)", type=["pt", "onnx"], key=f"ai_roi_model_{source_hash[:10]}")
    ai_confidence = st.slider("Detection confidence", 0.05, 0.95, 0.35, 0.05, key=f"ai_conf_{source_hash[:10]}")
    ai_imgsz = st.selectbox("Model inference image size", [320, 480, 640, 800, 1024], index=2, key=f"ai_imgsz_{source_hash[:10]}")
    ai_norm = st.selectbox("Thermal-to-model normalization", ["1st–99th percentile", "Fixed numeric range"], key=f"ai_norm_{source_hash[:10]}")
    ai_fixed_min = ai_fixed_max = None
    if ai_norm == "Fixed numeric range":
        ai_fixed_min = st.number_input("Model input minimum", value=0.0, key=f"ai_min_{source_hash[:10]}")
        ai_fixed_max = st.number_input("Model input maximum", value=255.0, key=f"ai_max_{source_hash[:10]}")
    if ai_weights is not None:
        st.caption(f"Model file: {ai_weights.name} · {ai_weights.size / 1024**2:.1f} MiB")
        if st.button("Suggest ROIs on selected reference frame", key=f"ai_suggest_{source_hash[:10]}"):
            try:
                suggestions = suggest_rois_yolo_seg(
                    frames[editor_frame_idx], ai_weights.getvalue(), filename=ai_weights.name,
                    confidence=float(ai_confidence), image_size=int(ai_imgsz),
                    normalization="percentile" if ai_norm.startswith("1st") else "fixed",
                    fixed_min=ai_fixed_min, fixed_max=ai_fixed_max,
                    max_regions=max(1, 8-len(initial_regions)))
                if not suggestions:
                    st.warning("The model returned no segmentation regions for this frame.")
                else:
                    existing = extract_component_regions(component_result, initial_regions)
                    combined = validate_regions(existing + suggestions, (h,w))
                    st.session_state[region_session_key] = combined
                    st.success(f"Added {len(suggestions)} AI-suggested ROI(s). Review/edit them in the polygon editor before analysis.")
                    st.rerun()
            except Exception as exc:
                st.error(f"AI ROI suggestion failed: {exc}")
    else:
        st.info("AI segmentation is optional. Install the AI extra and upload a trusted thermal-domain YOLO-seg model. No pretrained perfusion model is bundled.")


with st.expander("03 · Select a reference ROI and feature for sensor comparison", expanded=False):
    sensor_roi_name = st.selectbox("ROI to compare with reference channels", [r["name"] for r in analysis_regions], key=f"sensor_roi_{source_hash[:10]}")
    sensor_feature = st.selectbox("Thermal feature", ["median_event_delta", "mean_event_delta", "median_delta_from_baseline", "mean_dT_dt", "median", "mean"],
        format_func=lambda v: {"median_event_delta":"Median event-relative Δ", "mean_event_delta":"Mean event-relative Δ",
                               "median_delta_from_baseline":"Median baseline-relative Δ", "mean_dT_dt":"Mean dT/dt",
                               "median":"ROI median signal", "mean":"ROI mean signal"}[v],
        key=f"sensor_feature_{source_hash[:10]}")
    max_lag_seconds = st.slider("Maximum exploratory lag search (seconds)", 0.0, 30.0, 5.0, 0.5, key=f"sensor_lag_{source_hash[:10]}")
    st.caption("Correlation and lag estimates are descriptive only. Sensor clocks must share a documented time origin or be corrected using the per-channel clock offset.")


# ----------------------------------------------------------------------------
# Cache signature and analysis action
# ----------------------------------------------------------------------------
settings = {
    "source_name": source_name, "source_hash": source_hash, "units": units, "calibrated_celsius": bool(calibrated),
    "fps": float(fps), "baseline_frames": int(baseline_frames), "protocol_mode": mode_id,
    "challenge_start": challenge_start, "challenge_end": challenge_end,
    "gap_known": bool(gap_known), "gap_seconds": gap_seconds,
    "timestamps_include_gap": bool(timestamps_include_gap),
    "timestamps_hash": hashlib.sha256(np.asarray(timestamp_values, dtype=float).tobytes()).hexdigest() if timestamp_values is not None else None,
    "threshold": float(threshold), "spatial_sigma": float(spatial_sigma), "temporal_sigma": float(temporal_sigma),
    "register": bool(register), "registration_mode": registration_mode,
    "pixel_size_mm": float(pixel_size_mm) if physical_scale_known else None,
    "ambient_c": float(ambient_c), "humidity": float(humidity), "acclimatization_min": float(acclimatization),
    "camera_distance_cm": float(camera_distance), "emissivity": float(emissivity),
    "reflected_temperature_c": float(reflected_temperature), "transmissivity": float(transmissivity),
    "window_temperature_c": float(window_temperature), "camera_profile_id": camera_profile_id,
    "camera_profile_label": selected_profile_label, "camera_connection": camera_connection,
    "camera_form_factor": camera_form_factor, "detector_architecture": detector_architecture,
    "spectral_band": spectral_band, "detector_output_type": detector_output_type,
    "physical_interface": physical_interface, "custom_vendor_name": custom_vendor_name,
    "calibration_state": calibration_state, "pixel_format_label": pixel_format_label,
    "camera_model": camera_model, "camera_serial": camera_serial, "calibration_id": calibration_id,
    "camera_adapter_installed": bool(camera_profile_status["plugin_installed"]),
    "camera_parameter_values": _redact_secrets(camera_parameter_values),
    "detected_camera": detected_camera,
    "vendor_metadata": vendor_metadata,
    "reference_signals": {k: {kk: vv for kk, vv in v.items() if kk != "payload"} for k, v in sensor_uploads.items()},
    "sensor_roi_name": sensor_roi_name, "sensor_feature": sensor_feature, "max_lag_seconds": float(max_lag_seconds),
    "notes": notes, "regions": regions,
    "front_reference_frame": int(front_reference_frame),
    "front_reference_frames": int(event_reference_frames),
}
signature = hashlib.sha256(json.dumps(_clean(settings), sort_keys=True).encode()).hexdigest()

if timeline.warnings:
    with st.expander("Protocol and timing notices", expanded=False):
        for warning in timeline.warnings: st.warning(warning)

if run:
    try:
        with st.status("Analysing full-resolution thermal sequence…", expanded=True) as status:
            status.write("Preprocessing: denoising and optional image registration")
            processed, prep_meta = preprocess_sequence(frames, spatial_sigma=float(spatial_sigma),
                                                       register=bool(register),
                                                       registration_reference_frames=int(baseline_frames),
                                                       registration_mode=registration_mode)
            status.write("Computing baseline, Δ maps, temporal derivatives and apparent front-arrival maps")
            union = np.logical_or.reduce(list(masks.values())) if masks else np.ones((h,w), dtype=bool)
            result = analyze_sequence(
                processed, times_s=timeline.times_s, roi_mask=union,
                baseline_frames=int(baseline_frames), change_threshold=float(threshold),
                pixel_size_mm=float(pixel_size_mm) if physical_scale_known else None,
                temporal_smoothing_sigma=float(temporal_sigma),
                derivative_breaks=timeline.derivative_breaks,
                front_reference_frame=int(front_reference_frame),
                front_reference_frames=int(event_reference_frames),
            )
            result.metrics["phase"] = timeline.phase_labels
            result.summary["sequence_name"] = source_name
            result.summary["units"] = units
            result.summary["preprocessing"] = prep_meta
            result.summary["protocol"] = timeline.metadata
            result.summary["phase_spans"] = timeline.phase_spans
            result.summary["acquisition_metadata"] = {
                "ambient_temperature_c": float(ambient_c), "relative_humidity_percent": float(humidity),
                "acclimatization_minutes": float(acclimatization), "camera_distance_cm": float(camera_distance),
                "emissivity": float(emissivity), "reflected_temperature_c": float(reflected_temperature),
                "transmissivity": float(transmissivity), "window_temperature_c": float(window_temperature),
                "camera_vendor_profile": camera_profile_id, "camera_profile_label": selected_profile_label,
                "camera_connection_workflow": camera_connection, "camera_model": camera_model,
                "camera_serial": camera_serial, "calibration_id": calibration_id,
                "vendor_metadata": vendor_metadata,
                "radiometric_calibration_confirmed": bool(calibrated), "pixel_value_units": units,
                "notes": notes,
            }
            regional_metrics = {}
            regional_fits = {}
            recovery_start = int(baseline_frames) if mode_id == "baseline_recovery_unrecorded_challenge" else int(challenge_end)
            for region in analysis_regions:
                mask = masks[region["id"]]
                df = roi_metrics_from_result(result, mask, region["name"])
                df["phase"] = timeline.phase_labels
                regional_metrics[region["name"]] = df
                if n - recovery_start >= 10:
                    regional_fits[region["name"]] = fit_rewarming_time_constant(
                        df["time_s"].to_numpy(), df["median"].to_numpy(), start_index=recovery_start)
                else:
                    regional_fits[region["name"]] = {"valid": False, "reason": "Fewer than 10 samples after the configured recovery start."}
            reference_analysis = {}
            reference_timeseries = {}
            if sensor_uploads:
                for sensor_name, cfg in sensor_uploads.items():
                    modality = cfg.get("modality", sensor_name.split(" ")[0].casefold())
                    signal = parse_reference_csv(
                        cfg["payload"], cfg["time_column"], cfg["value_column"], name=sensor_name,
                        modality=modality, units=cfg["units"], source_name=cfg["filename"],
                        time_offset_s=cfg["time_offset_s"], processing=cfg["processing"],
                    )
                    thermal_df = regional_metrics[sensor_roi_name]
                    sensor_stats, sensor_frame = compare_reference_signal(
                        signal, timeline.times_s, thermal_df[sensor_feature].to_numpy(dtype=float),
                        max_lag_s=float(max_lag_seconds), derivative_breaks=timeline.derivative_breaks)
                    reference_analysis[sensor_name] = sensor_stats
                    reference_timeseries[sensor_name] = sensor_frame
            result.summary["reference_signal_analysis"] = reference_analysis
            result.summary["regional_recovery_fits"] = regional_fits
            result.summary["regional_roi_pixels"] = {r["name"]: int(masks[r["id"]].sum()) for r in analysis_regions}
            result.summary["roi_overlap"] = overlap
            qc = compute_quality_report(frames, timeline.times_s, int(baseline_frames), union,
                np.asarray(prep_meta.get("registration_shifts_yx_px", np.zeros((n,2)))),
                float(overlap["overlap_fraction_of_union"]))
            status.write("Preparing figures and reproducible export bundle")
            png = _figure_png(result, processed, units)
            display_frames = result.delta
            player, player_meta = make_frame_player(
                result.event_delta, timeline.times_s, None, view_mode="Event progression", unit_label=units,
                playback_speed=1.0, roi_regions=analysis_regions, phase_labels=timeline.phase_labels,
            )
            player_html = player.to_html(full_html=True, include_plotlyjs="cdn")
            export_settings = dict(settings)
            export_settings["regional_recovery_fits"] = regional_fits
            bundle, export_summary = _export_bundle(result, processed, regional_metrics, analysis_regions,
                                                     timeline, qc, export_settings, png, player_html,
                                                     reference_analysis=reference_analysis,
                                                     reference_timeseries=reference_timeseries)
            st.session_state["tp_analysis"] = {
                "signature": signature, "result": result, "processed": processed,
                "regions": analysis_regions, "masks": masks, "regional_metrics": regional_metrics,
                "regional_fits": regional_fits, "timeline": timeline, "qc": qc,
                "summary_png": png, "player_html": player_html, "bundle": bundle,
                "summary": export_summary, "player_meta": player_meta, "settings": export_settings,
                "reference_analysis": reference_analysis, "reference_timeseries": reference_timeseries,
            }
            status.update(label="Analysis complete", state="complete", expanded=False)
        st.success("Analysis completed. Explore the player, propagation maps, ROI trends, QC and exports below.")
    except Exception as exc:
        st.exception(exc)

analysis = st.session_state.get("tp_analysis")
if analysis is None:
    st.info("Configure the acquisition and ROI annotations, then click **Run full analysis** in the sidebar.")
    st.stop()
if analysis.get("signature") != signature:
    st.warning("Your input or settings have changed since the previous run. The displayed results are stale; click **Run full analysis** again to regenerate them.")
    st.stop()

result = analysis["result"]
processed = analysis["processed"]
regions = analysis["regions"]
regional_metrics = analysis["regional_metrics"]
regional_fits = analysis["regional_fits"]
qc = analysis["qc"]
timeline = analysis["timeline"]
summary = result.summary
reference_analysis = analysis.get("reference_analysis", {})
reference_timeseries = analysis.get("reference_timeseries", {})


# ----------------------------------------------------------------------------
# Workspace summary and tabs
# ----------------------------------------------------------------------------
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Frames analysed", f"{n:,}")
k2.metric("ROI coverage", f"{union_mask.mean():.1%}")
k3.metric("Detected warming", f"{summary['warming_event_fraction']:.1%}")
k4.metric("Detected cooling", f"{summary['cooling_event_fraction']:.1%}")
k5.metric("Applied threshold", f"{summary['applied_change_threshold']:.4g} {units}")

player_tab, maps_tab, roi_tab, qc_tab, integrations_tab, export_tab = st.tabs([
    "🎞️ Frame explorer", "🗺️ Propagation maps", "📈 ROI dynamics", "🧪 Protocol & QC",
    "🧠 AI & sensor integrations", "⬇️ Export"
])

with player_tab:
    st.markdown("<div class='tp-section'><span class='tp-live'></span>Animated thermal sequence explorer</div>", unsafe_allow_html=True)
    view_mode = st.segmented_control("Display channel", ["Thermal sequence", "Δ from baseline", "Event progression", "Rate of change"],
                                     default="Δ from baseline", key="view_mode")
    speed = st.select_slider("Playback speed", options=[0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0], value=1.0,
                             format_func=lambda x: f"{x:g}×")
    display_frames = (processed if view_mode == "Thermal sequence" else result.delta if view_mode == "Δ from baseline"
                      else result.event_delta if view_mode == "Event progression" else result.delta_rate)
    fig, player_meta = make_frame_player(display_frames, timeline.times_s, None, view_mode=view_mode,
        unit_label=units, playback_speed=float(speed), roi_regions=regions, phase_labels=timeline.phase_labels)
    st.plotly_chart(fig, width="stretch", config={"displaylogo":False,"scrollZoom":True,"responsive":True},
                    key=f"tp_player_{view_mode.replace(' ','_')}")
    st.caption(f"Browser preview: {player_meta['preview_frames']} frames · stride {player_meta['spatial_stride']} px. Numerical processing retains all {n:,} frames at {w} × {h}.")
    st.caption("The Plotly player runs in the browser. Play/Pause and the frame slider do not rerun the analysis pipeline.")
    with st.expander("Selected frame inspection", expanded=False):
        selected_frame = st.slider("Frame index", 0, n-1, min(int(baseline_frames), n-1), key="selected_frame_static")
        frame = display_frames[selected_frame]
        fig_frame = go.Figure(go.Heatmap(z=frame, colorscale="Inferno" if view_mode=="Thermal sequence" else "Turbo" if view_mode=="Rate of change" else "RdBu_r"))
        for region in regions:
            pts=np.asarray(region["points"],dtype=float)
            fig_frame.add_trace(go.Scatter(x=np.r_[pts[:,0],pts[0,0]],y=np.r_[pts[:,1],pts[0,1]],
                mode="lines",line={"color":region["color"],"width":2},name=region["name"]))
        fig_frame.update_layout(template="plotly_dark",height=530,margin={"l":0,"r":0,"t":25,"b":5},
                                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        fig_frame.update_yaxes(autorange="reversed",scaleanchor="x",scaleratio=1)
        st.plotly_chart(fig_frame,width="stretch",config={"displaylogo":False,"scrollZoom":True})
        st.caption(f"Frame {selected_frame} · {timeline.times_s[selected_frame]:.3f} s · phase: {timeline.phase_labels[selected_frame]}")

with maps_tab:
    st.markdown("<div class='tp-section'>Apparent thermal-front propagation</div>", unsafe_allow_html=True)
    a,b=st.columns(2)
    for col, label, key in [(a,"Warming front","warming_propagation"),(b,"Cooling front","cooling_propagation")]:
        d=summary[key]
        with col:
            st.markdown(f"<div class='tp-card'><div class='tp-label'>{label}</div><div class='tp-value'>{d.get('direction','insufficient detected front')}</div><div class='tp-value tp-mono'>{d.get('median_front_speed',float('nan')):.3g} {d.get('speed_unit','px/s')}</div><div class='tp-note'>Median apparent normal speed · direction coherence {d.get('direction_coherence_0_to_1',float('nan')):.2f}</div></div>",unsafe_allow_html=True)
    st.caption(f"Front-arrival maps are measured relative to frame {summary['front_reference_frame'] + 1} (median of {summary['front_reference_frames']} reference frame(s)). Baseline-relative Δ is shown separately. Arrow direction follows later threshold arrival; it is not a vascular map or blood-flow velocity.")
    st.plotly_chart(make_arrival_figure(result),width="stretch",config={"displaylogo":False,"scrollZoom":True})
    st.markdown("**Threshold sensitivity**")
    threshold_rows=[]
    base_thr=float(summary["applied_change_threshold"])
    for factor in [0.75,1.0,1.25,1.5]:
        warm=(result.event_delta>=base_thr*factor).any(axis=0)&result.roi_mask
        cool=(result.event_delta<=-base_thr*factor).any(axis=0)&result.roi_mask
        threshold_rows.append({"Threshold multiplier":factor,"Warming coverage":float(warm.sum()/result.roi_mask.sum()),
                               "Cooling coverage":float(cool.sum()/result.roi_mask.sum())})
    sens=pd.DataFrame(threshold_rows)
    st.dataframe(sens.style.format({"Warming coverage":"{:.1%}","Cooling coverage":"{:.1%}"}),width="stretch",hide_index=True)
    st.caption("Coverage sensitivity describes how many ROI pixels cross each alternative threshold at any observed time. It is not a validated accuracy metric.")

with roi_tab:
    st.markdown("<div class='tp-section'>Named region time-series and recovery behaviour</div>",unsafe_allow_html=True)
    metric_labels={"ROI median":"median","ROI mean":"mean","Spatial standard deviation":"std",
                   "Median Δ from baseline":"median_delta_from_baseline","Mean Δ from baseline":"mean_delta_from_baseline",
                   "Median Δ from event reference":"median_event_delta","Mean Δ from event reference":"mean_event_delta",
                   "Mean dT/dt":"mean_dT_dt","Warming fraction from baseline":"warming_fraction",
                   "Cooling fraction from baseline":"cooling_fraction","Warming fraction from event reference":"event_warming_fraction",
                   "Cooling fraction from event reference":"event_cooling_fraction"}
    selected_label=st.selectbox("Feature to compare",list(metric_labels))
    selected_metric=metric_labels[selected_label]
    roi_fig=make_named_roi_figure(regional_metrics,selected_metric,units,timeline.phase_spans,timeline.time_axis_label)
    st.plotly_chart(roi_fig,width="stretch",config={"displaylogo":False,"scrollZoom":True})
    fit_rows=[]
    for name,fit in regional_fits.items():
        fit_rows.append({"ROI":name,"Fit valid":bool(fit.get("valid")),"Tau (s)":fit.get("tau_s"),
                         "R²":fit.get("r_squared"),"Reason / note":fit.get("reason",fit.get("note",""))})
    st.markdown("**Descriptive exponential recovery fits**")
    st.dataframe(pd.DataFrame(fit_rows),width="stretch",hide_index=True)
    st.caption("The fitted time constant describes the shape of the ROI thermal recovery curve. It is not a direct or validated perfusion measurement.")
    with st.expander("Regional frame metrics"):
        for name,df in regional_metrics.items():
            st.markdown(f"**{name}**")
            st.dataframe(df,width="stretch",height=230,hide_index=True)

    with st.expander("Exploratory spectral / time-frequency analysis", expanded=False):
        spec_roi = st.selectbox("ROI for spectral analysis", list(regional_metrics), key="spectral_roi")
        spec_choices = ["median_event_delta", "mean_event_delta", "mean_dT_dt", "median", "mean"]
        spec_feature = st.selectbox("ROI signal", spec_choices,
            format_func=lambda v: {"median_event_delta":"Median event-relative Δ", "mean_event_delta":"Mean event-relative Δ",
                "mean_dT_dt":"Mean dT/dt", "median":"ROI median temperature/intensity", "mean":"ROI mean temperature/intensity"}[v],
            key="spectral_feature")
        max_freq = float(0.5 / np.median(np.diff(timeline.times_s))) if n > 1 else 0.0
        spectral_method = st.radio("Method", ["Welch power spectrum", "Continuous wavelet scalogram"], horizontal=True, key="spectral_method")
        if max_freq <= 0.005:
            st.warning("The current sampling interval leaves too little frequency bandwidth for this spectral viewer.")
        else:
            spectral_limit = st.number_input("Maximum frequency (Hz)", min_value=0.005, max_value=max_freq,
                value=max_freq, step=max(0.005, max_freq/20), key="spectral_max_freq")
            if st.button("Compute exploratory spectrum", key="compute_spectral"):
                try:
                    sdf = regional_metrics[spec_roi]
                    values = sdf[spec_feature].to_numpy(dtype=float)
                    if spectral_method.startswith("Welch"):
                        freq, power, meta = welch_spectrum(timeline.times_s, values,
                            derivative_breaks=timeline.derivative_breaks, max_frequency_hz=float(spectral_limit))
                        st.session_state["tp_spectral_result"] = {"kind":"welch", "frequency":freq, "power":power, "meta":meta}
                    else:
                        wt, freq, power, meta = wavelet_scalogram(timeline.times_s, values,
                            derivative_breaks=timeline.derivative_breaks, max_frequency_hz=float(spectral_limit))
                        st.session_state["tp_spectral_result"] = {"kind":"cwt", "time":wt, "frequency":freq, "power":power, "meta":meta}
                except Exception as exc:
                    st.error(str(exc))
        spectral = st.session_state.get("tp_spectral_result")
        if spectral:
            meta = spectral["meta"]
            st.json(meta, expanded=False)
            if spectral["kind"] == "welch":
                fig_psd = go.Figure(go.Scatter(x=spectral["frequency"], y=spectral["power"], mode="lines", name="PSD"))
                fig_psd.update_layout(template="plotly_dark", height=330, xaxis_title="Frequency (Hz)", yaxis_title="Power spectral density",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            else:
                fig_psd = go.Figure(go.Heatmap(x=spectral["time"], y=spectral["frequency"],
                    z=np.log10(np.maximum(spectral["power"], 1e-12)), colorscale="Turbo",
                    colorbar={"title":"log10 power"}))
                fig_psd.update_layout(template="plotly_dark", height=420, xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_psd, width="stretch", config={"displaylogo":False})
            st.warning("Thermal spectra are camera- and protocol-dependent. Check frame rate, segment duration, sensor sensitivity and Nyquist limit; frequency peaks are not automatically assigned to vascular mechanisms. CWT requires the optional wavelet dependency.")

with qc_tab:
    st.markdown("<div class='tp-section'>Acquisition protocol and quality indicators</div>",unsafe_allow_html=True)
    cols=st.columns(6)
    cols[0].metric("Finite samples",f"{qc['finite_sample_fraction']:.2%}")
    cols[1].metric("Frame interval CV",f"{qc['frame_interval_cv']:.2%}")
    cols[2].metric("Baseline drift",f"{qc['baseline_median_drift_per_min_input_units']:.4g} /min")
    cols[3].metric("Max translation",f"{qc['max_registration_shift_px']:.2f} px")
    cols[4].metric("3×IQR outliers",f"{qc['sampled_outlier_pixel_fraction_3_iqr']:.2%}")
    cols[5].metric("ROI overlap",f"{qc['roi_overlap_fraction_of_union']:.2%}")
    st.markdown("**Protocol phases**")
    st.dataframe(pd.DataFrame(timeline.phase_spans),width="stretch",hide_index=True)
    st.write(f"Time axis: **{timeline.time_axis_label}**")
    for warning in [*timeline.warnings,*qc["warnings"]]: st.warning(warning)
    if not timeline.warnings and not qc["warnings"]: st.success("No threshold-triggered engineering QC warnings were raised. This is not a clinical quality certification.")
    if register:
        shifts=np.asarray(summary["preprocessing"].get("registration_shifts_yx_px",np.zeros((n,2))),dtype=float)
        st.write(f"Registration model: **{registration_mode_label}**")
        st.plotly_chart(make_motion_figure(shifts),width="stretch",config={"displaylogo":False})
    st.markdown("**Recorded acquisition metadata**")
    st.json(summary.get("acquisition_metadata",{}),expanded=False)
    st.markdown("**Protocol checklist**")
    checklist=["Allow subject and camera to acclimatize in a stable room.",
               "Preserve original radiometric export and calibration settings.",
               "Record ambient temperature, humidity, camera distance, emissivity and frame rate.",
               "Keep camera geometry and subject posture stable.",
               "Document challenge start/end and distinguish observed from unrecorded phases.",
               "For perfusion claims, acquire a synchronized independent reference such as LSCI or laser-Doppler."]
    for item in checklist: st.markdown(f"- {item}")

with integrations_tab:
    st.markdown("<div class='tp-section'>Optional integrations and research AI</div>", unsafe_allow_html=True)
    st.caption("Integrations are independently selectable. A missing vendor SDK or optional AI dependency does not block core thermal analysis.")
    cam_status = adapter_status(camera_profile_id)
    c1, c2, c3 = st.columns(3)
    c1.metric("Camera profile", selected_profile_label)
    c2.metric("Adapter plugin", "Available" if cam_status["plugin_installed"] else "Not installed")
    c3.metric("Reference channels", str(len(reference_analysis)))
    with st.expander("Camera integration capability details", expanded=False):
        st.json(cam_status, expanded=False)
        st.write("Native proprietary decoding and live acquisition require the matching vendor SDK plus a thermoprop adapter plugin. Choosing a camera profile records the intended hardware; it does not itself establish a connection.")
        st.markdown("Vendor documentation: " + (f"[Open SDK docs]({cam_status['documentation_url']})" if cam_status["documentation_url"] else "No vendor URL supplied."))

    st.markdown("### Synchronized physiological reference signals")
    if reference_analysis:
        ref_rows = []
        for name, item in reference_analysis.items():
            ref_rows.append({"Reference": name, "Modality": item.get("modality"), "N aligned": item.get("n_aligned"),
                "Coverage": item.get("coverage_fraction"), "Pearson r": item.get("pearson_r"),
                "Spearman rho": item.get("spearman_rho"), "Best lag (s)": item.get("best_lag_s"), "Status": item.get("status")})
        ref_df = pd.DataFrame(ref_rows)
        st.dataframe(ref_df, width="stretch", hide_index=True)
        st.warning("These are exploratory, serially dependent signal associations. Uncorrected p-values are not confirmatory. LSCI / laser-Doppler are the preferred independent local perfusion references; PPG, ECG and EDA are complementary signals.")
        for name, frame_df in reference_timeseries.items():
            st.markdown(f"**{name} — aligned time series**")
            fig_ref = go.Figure()
            fig_ref.add_trace(go.Scatter(x=frame_df["time_s"], y=frame_df["thermal_feature"], name=f"Thermal · {sensor_feature}", yaxis="y1"))
            fig_ref.add_trace(go.Scatter(x=frame_df["time_s"], y=frame_df["reference_signal"], name=name, yaxis="y2"))
            fig_ref.update_layout(template="plotly_dark", height=340, margin={"l":10,"r":10,"t":30,"b":10},
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis={"title":"Thermal feature"}, yaxis2={"title":"Reference signal", "overlaying":"y", "side":"right"},
                legend={"orientation":"h", "y":1.08})
            st.plotly_chart(fig_ref, width="stretch", config={"displaylogo":False})
            st.download_button(f"Download {name} aligned CSV", frame_df.to_csv(index=False).encode(),
                               f"{name.lower().replace(' ','_')}_aligned.csv", "text/csv", key=f"download_ref_{name}")
    else:
        st.info("No independent reference signal was uploaded. Add LSCI or laser-Doppler data for a meaningful perfusion-reference comparison. PPG/ECG/GSR can add complementary context but are not local perfusion ground truth.")
    st.download_button("Download reference-signal CSV template", signal_template_csv(),
                       "reference_signal_template.csv", "text/csv")

    st.markdown("### Participant-level feature export")
    st.caption("Create one row per ROI for a cohort training table. Fill in participant/session IDs and the independently measured reference target before combining rows from multiple participants.")
    participant_id = st.text_input("Participant identifier for this export", value="FILL_IN", key=f"feature_participant_{source_hash[:10]}")
    session_id = st.text_input("Session identifier for this export", value="FILL_IN", key=f"feature_session_{source_hash[:10]}")
    participant_features = build_participant_feature_table(
        regional_metrics, regional_fits, timeline.phase_labels,
        summary.get("acquisition_metadata", {}), participant_id=participant_id, session_id=session_id)
    st.dataframe(participant_features, width="stretch", hide_index=True)
    st.download_button("Download participant / ROI feature rows", participant_features.to_csv(index=False).encode(),
        "participant_roi_feature_rows.csv", "text/csv")
    st.caption("Each row is a participant/session/ROI summary, not a video frame. Add the corresponding LSCI or laser-Doppler reference value in `reference_perfusion` and group all sessions from one person under the same `participant_id`.")

    st.markdown("### Supervised AI model studio")
    st.write("Train an exploratory regression model only from paired participant/session/ROI summaries and independently measured reference targets. This does not train on frames and does not invent perfusion labels.")
    st.download_button("Download labelled feature-table template", feature_table_template(),
                       "thermal_reference_training_template.csv", "text/csv")
    with st.expander("Train grouped reference model", expanded=False):
        st.warning("Use one row per participant/session/ROI summary. Do not upload every frame as a row. At least six distinct participant groups are required; independent external-cohort validation is still necessary.")
        model_table_file = st.file_uploader("Paired thermal-feature + reference-label CSV", type=["csv"], key=f"model_table_{source_hash[:10]}")
        if model_table_file is not None:
            try:
                model_table = pd.read_csv(model_table_file)
                string_cols = list(model_table.select_dtypes(exclude=[np.number]).columns)
                number_cols = list(model_table.select_dtypes(include=[np.number]).columns)
                group_default = "participant_id" if "participant_id" in model_table.columns else (string_cols[0] if string_cols else model_table.columns[0])
                target_default = "reference_perfusion" if "reference_perfusion" in number_cols else (number_cols[0] if number_cols else model_table.columns[-1])
                group_col = st.selectbox("Participant grouping column", model_table.columns.tolist(),
                    index=model_table.columns.get_loc(group_default), key=f"model_group_{source_hash[:10]}")
                target_options = [c for c in number_cols if c != group_col]
                if not target_options:
                    st.error("The uploaded table needs at least one numeric reference-target column separate from the participant grouping column.")
                else:
                    target_col = st.selectbox("Reference target column", target_options,
                        index=target_options.index(target_default) if target_default in target_options else 0,
                        key=f"model_target_{source_hash[:10]}")
                    feature_options = [c for c in number_cols if c not in {group_col, target_col}]
                    chosen_features = st.multiselect("Thermal / acquisition feature columns", feature_options,
                        default=feature_options, key=f"model_features_{source_hash[:10]}")
                    estimator = st.selectbox("Model", ["ElasticNet", "Random Forest", "Extra Trees", "HistGradientBoosting", "XGBoost"],
                        index=1, key=f"model_kind_{source_hash[:10]}")
                    if st.button("Train with participant-grouped cross-validation", key=f"train_model_{source_hash[:10]}"):
                        model_eval = train_grouped_reference_model(model_table, target_column=target_col,
                            group_column=group_col, model_name=estimator, feature_columns=chosen_features)
                        usable = model_table.copy()
                        usable[target_col] = pd.to_numeric(usable[target_col], errors="coerce")
                        usable[group_col] = usable[group_col].astype(str).str.strip()
                        usable = usable[usable[target_col].notna() & usable[group_col].ne("")]
                        st.session_state["tp_reference_model_eval"] = {
                            "metrics": model_eval.metrics, "predictions": model_eval.predictions,
                            "importance": model_eval.feature_importance, "warnings": model_eval.warnings,
                            "model_name": model_eval.model_name, "features": model_eval.feature_columns,
                            "fitted_model": model_eval.fitted_model,
                            "feature_matrix": usable[model_eval.feature_columns].apply(pd.to_numeric, errors="coerce"),
                        }
            except Exception as exc:
                st.error(f"Model training input error: {exc}")
        model_eval_state = st.session_state.get("tp_reference_model_eval")
        if model_eval_state:
            m = model_eval_state["metrics"]
            mcols = st.columns(4)
            mcols[0].metric("Participants", f"{int(m['participants'])}")
            mcols[1].metric("Grouped OOF MAE", f"{m['mae']:.4g}")
            mcols[2].metric("Grouped OOF RMSE", f"{m['rmse']:.4g}")
            mcols[3].metric("Grouped OOF R²", f"{m['r2']:.3f}")
            st.dataframe(model_eval_state["importance"], width="stretch", hide_index=True)
            st.dataframe(model_eval_state["predictions"], width="stretch", hide_index=True)
            if model_eval_state["model_name"] in {"Random Forest", "Extra Trees", "XGBoost"}:
                with st.expander("Optional explainability with SHAP", expanded=False):
                    st.caption("SHAP describes how the fitted tree model uses input features on this cohort. It is not causal explanation and does not replace independent external validation.")
                    if st.button("Calculate global SHAP feature contributions", key="run_shap_reference_model"):
                        try:
                            shap_frame = explain_tree_model(model_eval_state["fitted_model"],
                                model_eval_state["feature_matrix"], max_samples=min(256, len(model_eval_state["feature_matrix"])))
                            st.session_state["tp_reference_model_shap"] = shap_frame
                        except Exception as exc:
                            st.error(f"SHAP explanation unavailable: {exc}")
                    shap_frame = st.session_state.get("tp_reference_model_shap")
                    if shap_frame is not None:
                        st.dataframe(shap_frame, width="stretch", hide_index=True)
                        shap_fig = go.Figure(go.Bar(x=shap_frame["mean_absolute_shap"], y=shap_frame["feature"], orientation="h"))
                        shap_fig.update_layout(template="plotly_dark", height=max(300, 36*len(shap_frame)),
                            xaxis_title="Mean |SHAP value|", yaxis_title="Feature", yaxis={"categoryorder":"total ascending"})
                        st.plotly_chart(shap_fig, width="stretch")
                        st.download_button("Download SHAP importance CSV", shap_frame.to_csv(index=False).encode(),
                            "shap_feature_importance.csv", "text/csv", key="download_shap_importance")
            else:
                st.info("SHAP global tree explanations are available for Random Forest, Extra Trees and XGBoost. Choose one of these estimators to enable this optional panel.")
            for warning in model_eval_state["warnings"]: st.warning(warning)

    st.markdown("### Physics-informed thermal modelling")
    st.caption("Explore model assumptions separately from measured thermal analysis. These forward-model outputs are not estimates of an individual's blood flow.")
    with st.expander("Pennes bioheat forward simulator", expanded=False):
        st.info("This is an idealized one-dimensional homogeneous tissue-slab forward simulation. It shows how an assumed perfusion parameter changes a modelled depth/time temperature field. It is not an inverse solver and must not be presented as measured or inferred perfusion.")
        pcols = st.columns(3)
        sim_duration = pcols[0].number_input("Simulation duration (s)", min_value=20.0, max_value=900.0, value=120.0, step=10.0, key="pennes_duration")
        sim_output_dt = pcols[1].select_slider("Output interval (s)", options=[0.25, 0.5, 1.0, 2.0, 5.0], value=1.0, key="pennes_output_dt")
        sim_depth = pcols[2].number_input("Model depth (mm)", min_value=2.0, max_value=30.0, value=10.0, step=1.0, key="pennes_depth")
        pcols2 = st.columns(3)
        perf_low = pcols2[0].number_input("Low assumed perfusion (s⁻¹)", min_value=0.0, max_value=0.02, value=0.0001, format="%.5f", key="pennes_perf_low")
        perf_nom = pcols2[1].number_input("Nominal assumed perfusion (s⁻¹)", min_value=0.0, max_value=0.02, value=0.0005, format="%.5f", key="pennes_perf_nom")
        perf_high = pcols2[2].number_input("High assumed perfusion (s⁻¹)", min_value=0.0, max_value=0.02, value=0.0010, format="%.5f", key="pennes_perf_high")
        pcols3 = st.columns(3)
        sim_ambient = pcols3[0].number_input("Ambient temperature (°C)", value=float(ambient_c), step=0.5, key="pennes_ambient")
        sim_challenge_ambient = pcols3[1].number_input("Challenge ambient (°C)", value=12.0, step=0.5, key="pennes_challenge_ambient")
        sim_arterial = pcols3[2].number_input("Assumed arterial temperature (°C)", value=37.0, step=0.5, key="pennes_arterial")
        pcols4 = st.columns(2)
        sim_start_default = min(20.0, float(sim_duration) * 0.2)
        sim_start = pcols4[0].number_input("Challenge start (s)", min_value=0.0, max_value=float(sim_duration), value=sim_start_default, step=1.0, key="pennes_start")
        sim_end = pcols4[1].number_input("Challenge end (s)", min_value=float(sim_start), max_value=float(sim_duration), value=min(max(40.0, sim_start), float(sim_duration)), step=1.0, key="pennes_end")
        with st.expander("Advanced tissue and boundary parameters", expanded=False):
            adv1 = st.columns(3)
            sim_initial = adv1[0].number_input("Initial tissue temperature (°C)", value=32.0, step=0.5, key="pennes_initial")
            sim_deep = adv1[1].number_input("Deep boundary temperature (°C)", value=36.5, step=0.5, key="pennes_deep")
            sim_surface_h = adv1[2].number_input("Surface heat-transfer coefficient (W/m²·K)", min_value=0.0, max_value=100.0, value=8.0, step=0.5, key="pennes_h")
            adv2 = st.columns(3)
            sim_density = adv2[0].number_input("Tissue density (kg/m³)", min_value=500.0, max_value=2000.0, value=1050.0, step=10.0, key="pennes_density")
            sim_specific_heat = adv2[1].number_input("Tissue specific heat (J/kg·K)", min_value=1000.0, max_value=6000.0, value=3600.0, step=50.0, key="pennes_cp")
            sim_conductivity = adv2[2].number_input("Tissue conductivity (W/m·K)", min_value=0.05, max_value=2.0, value=0.45, step=0.05, key="pennes_k")
            sim_metabolic = st.number_input("Metabolic heat source (W/m³)", min_value=0.0, max_value=5000.0, value=500.0, step=50.0, key="pennes_metabolic")
        if perf_low > perf_nom or perf_nom > perf_high:
            st.warning("For a useful sensitivity comparison, set low ≤ nominal ≤ high assumed perfusion.")
        if st.button("Run Pennes sensitivity simulation", key="run_pennes"):
            try:
                runs = {}
                for label, perf in [("Low", perf_low), ("Nominal", perf_nom), ("High", perf_high)]:
                    tt, depth, field, meta = simulate_pennes_slab(
                        duration_s=float(sim_duration), output_interval_s=float(sim_output_dt), depth_mm=float(sim_depth),
                        initial_temperature_c=float(sim_initial), deep_temperature_c=float(sim_deep),
                        arterial_temperature_c=float(sim_arterial), ambient_temperature_c=float(sim_ambient),
                        challenge_ambient_c=float(sim_challenge_ambient), challenge_start_s=float(sim_start),
                        challenge_end_s=float(sim_end), perfusion_s_inv=float(perf),
                        tissue_density_kg_m3=float(sim_density), tissue_specific_heat_j_kg_k=float(sim_specific_heat),
                        tissue_conductivity_w_m_k=float(sim_conductivity),
                        surface_heat_transfer_w_m2_k=float(sim_surface_h), metabolic_heat_w_m3=float(sim_metabolic))
                    runs[label] = {"time":tt, "depth":depth, "field":field, "metadata":meta}
                st.session_state["tp_pennes_runs"] = runs
            except Exception as exc:
                st.error(f"Pennes simulation error: {exc}")
        pennes_runs = st.session_state.get("tp_pennes_runs")
        if pennes_runs:
            sim_tab1, sim_tab2 = st.tabs(["Surface curves", "Depth × time"] )
            with sim_tab1:
                sim_fig = go.Figure()
                for label, run in pennes_runs.items():
                    sim_fig.add_trace(go.Scatter(x=run["time"], y=run["field"][:, 0], mode="lines", name=f"{label} perfusion"))
                sim_fig.update_layout(template="plotly_dark", height=360, xaxis_title="Time (s)", yaxis_title="Modelled surface temperature (°C)")
                st.plotly_chart(sim_fig, width="stretch")
            with sim_tab2:
                selected_run = st.selectbox("Sensitivity scenario", list(pennes_runs), key="pennes_selected_run")
                run = pennes_runs[selected_run]
                heat_fig = go.Figure(go.Heatmap(x=run["time"], y=run["depth"], z=run["field"].T, colorscale="Inferno", colorbar={"title":"°C"}))
                heat_fig.update_layout(template="plotly_dark", height=440, xaxis_title="Time (s)", yaxis_title="Depth (mm)")
                heat_fig.update_yaxes(autorange="reversed")
                st.plotly_chart(heat_fig, width="stretch")
                st.json(run["metadata"], expanded=False)
            st.download_button("Download Pennes scenario arrays (NPZ)",
                data=_npz_bytes({f"{name.lower()}_time_s":run["time"] for name,run in pennes_runs.items()} |
                    {f"{name.lower()}_depth_mm":run["depth"] for name,run in pennes_runs.items()} |
                    {f"{name.lower()}_temperature_c":run["field"] for name,run in pennes_runs.items()}),
                file_name="pennes_forward_sensitivity.npz", mime="application/octet-stream", key="download_pennes_npz")

    st.markdown("### Optional paper-derived heat-balance index")
    st.warning("Enabled only when the input is radiometrically calibrated in °C. This is the paper's model-derived heat-balance index under its assumptions, not a directly measured blood-flow map. Review the paper's assumptions and parameter sensitivity before interpretation.")
    with st.expander("Calculate Eq. (10) model index on the baseline map", expanded=False):
        if units != "°C" or not calibrated:
            st.info("Select °C and explicitly confirm radiometric calibration in the Acquisition panel before this model can be run.")
        else:
            eq_arterial = st.number_input("Assumed arterial temperature for Eq. (10) (°C)", min_value=30.0, max_value=42.0, value=39.0, step=0.1, key="eq10_arterial")
            eq_emissivity = st.number_input("Emissivity used by Eq. (10)", min_value=0.5, max_value=1.0, value=float(emissivity), step=0.01, key="eq10_emissivity")
            eq_exchange = st.number_input("Exchange ratio", min_value=0.1, max_value=2.0, value=0.8, step=0.05, key="eq10_exchange")
            if st.button("Calculate paper-derived heat-balance map", key="run_eq10"):
                try:
                    eq_map = wu_style_model_index(result.baseline, float(ambient_c),
                        arterial_temperature_c=float(eq_arterial), emissivity=float(eq_emissivity),
                        exchange_ratio=float(eq_exchange))
                    st.session_state["tp_eq10_map"] = {"map":eq_map, "params":{"arterial_temperature_c":eq_arterial,
                        "emissivity":eq_emissivity, "exchange_ratio":eq_exchange, "ambient_temperature_c":ambient_c}}
                except Exception as exc:
                    st.error(f"Heat-balance model error: {exc}")
            eq_state = st.session_state.get("tp_eq10_map")
            if eq_state and eq_state.get("source_hash") == source_hash:
                eq_values = eq_state["map"]
                finite_eq = np.isfinite(eq_values) & union_mask
                if finite_eq.any():
                    eq_fig = go.Figure(go.Heatmap(z=np.where(union_mask, eq_values, np.nan), colorscale="Viridis", colorbar={"title":"Model index"}))
                    eq_fig.update_layout(template="plotly_dark", height=420, title="Paper-derived Eq. (10) model index")
                    eq_fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
                    st.plotly_chart(eq_fig, width="stretch")
                    st.metric("Median model index", f"{np.nanmedian(eq_values[finite_eq]):.4g} kg/(m²·s)")
                    st.caption(f"Valid pixels: {int(finite_eq.sum()):,} · Parameters: {eq_state['params']}")
                    st.download_button("Download model-index map (NPY)", _npy_bytes(eq_values), "paper_heat_balance_model_index.npy", "application/octet-stream")
                else:
                    st.warning("No valid pixels for this set of model parameters (for example skin temperature at/below ambient or invalid denominator).")

    st.markdown("### Experiment tracking")
    st.write("MLflow can store this run's settings, summary and numeric metrics on a local or configured tracking server.")
    if st.button("Log this run to MLflow", key=f"mlflow_log_{source_hash[:10]}"):
        try:
            numeric_metrics = {
                "warming_coverage": float(summary.get("warming_event_fraction", 0.0)),
                "cooling_coverage": float(summary.get("cooling_event_fraction", 0.0)),
                "applied_threshold": float(summary.get("applied_change_threshold", 0.0)),
            }
            run_id = log_analysis_run(_clean(analysis["summary"]), _clean(analysis["settings"]), numeric_metrics)
            st.success(f"MLflow run logged: {run_id}")
        except Exception as exc:
            st.error(str(exc))
    st.caption("Optional install: pip install -e '.[tracking]'. Configure MLFLOW_TRACKING_URI if logging to a remote/local server.")

with export_tab:
    st.markdown("<div class='tp-section'>Reproducible analysis package</div>",unsafe_allow_html=True)
    st.write("The package contains the settings, named ROI geometry, phase timeline, regional metrics, analysis arrays, QC summary and an interactive frame-player HTML file.")
    st.download_button("⬇ Download union ROI metrics CSV",result.metrics.to_csv(index=False).encode(),
                       "union_roi_frame_metrics.csv","text/csv",width="stretch")
    region_csv=pd.concat(regional_metrics.values(),ignore_index=True).to_csv(index=False).encode()
    st.download_button("⬇ Download regional metrics CSV",region_csv,"regional_frame_metrics.csv","text/csv",width="stretch")
    summary_json=json.dumps(_clean(analysis["summary"]),indent=2,allow_nan=False).encode()
    st.download_button("⬇ Download summary JSON",summary_json,"summary.json","application/json",width="stretch")
    st.download_button("⬇ Download complete ZIP bundle",analysis["bundle"],"thermal_pattern_propagation_export.zip",
                       "application/zip",type="primary",width="stretch")
    st.image(analysis["summary_png"],caption="Static analysis summary included in export",width="stretch")
    st.caption("The original camera source file and timestamp file are not embedded in this export. Retain them separately with calibration documentation.")

st.markdown("---")
st.caption("Thermal Pattern Propagation Lab v1.5.0 · Research software · Not a medical device · No clinical perfusion validation is claimed.")
