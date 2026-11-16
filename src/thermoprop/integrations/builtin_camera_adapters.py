"""Built-in transport adapters that do not require a brand-specific thermography SDK.

GenTL acquires numeric camera components using the user's installed CTI producer.
It returns raw scalar samples and deliberately does not convert counts to Celsius.

HTTP Matrix API expects a user-configured endpoint returning JSON with a 3-D
`frames` array and optional `timestamps_s` / `metadata` fields. This is a generic
middleware contract, not an undocumented vendor endpoint implementation.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


def _device_value(info: Any, *names: str) -> str:
    for name in names:
        value = getattr(info, name, None)
        if value not in (None, ""):
            return str(value)
    return ""


class GenTLAdapter:
    """Generic GenICam/GenTL capture adapter backed by Harvester.

    The CTI producer must be installed separately. The adapter reads one 2-D
    scalar image component from each buffer, preserving the camera pixel format
    and timestamp. It does not assume that Mono16 values are temperatures.
    """

    def _harvester(self, settings: dict):
        try:
            from harvesters.core import Harvester
        except Exception as exc:
            raise RuntimeError(
                "GenTL support requires the optional vision dependencies. Install `pip install -e .[vision]`, "
                "then install the camera manufacturer's GenTL Producer (CTI file)."
            ) from exc
        cti = str((settings or {}).get("cti_file", "")).strip()
        if not cti:
            raise ValueError("Select the camera manufacturer's .cti GenTL Producer file first.")
        path = Path(cti).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"GenTL Producer file not found: {path}")
        h = Harvester()
        h.add_file(str(path), check_existence=True, check_validity=True)
        h.update()
        return h

    def get_parameter_schema(self, connection=None):
        if connection == "Recorded file":
            return []
        return [
            {"key": "cti_file", "label": "GenTL Producer (.cti) path", "type": "str", "default": "",
             "help": "Install the camera vendor's GenTL Producer and select its .cti file. The file is not bundled."},
            {"key": "component_index", "label": "Image component index", "type": "int", "default": 0, "min": 0, "max": 15, "step": 1,
             "help": "Usually 0 for a single-component thermal image. Inspect the camera payload documentation for multipart data."},
            {"key": "frame_timeout_s", "label": "Frame fetch timeout (s)", "type": "float", "default": 2.0, "min": 0.1, "max": 30.0, "step": 0.1},
            {"key": "apply_exposure_time", "label": "Set camera exposure time", "type": "bool", "default": False,
             "help": "Opt in only if the GenICam device documents an ExposureTime feature."},
            {"key": "exposure_time_us", "label": "Exposure time (µs)", "type": "float", "default": 5000.0, "min": 0.1, "max": 1000000.0, "step": 10.0},
            {"key": "apply_gain", "label": "Set camera gain", "type": "bool", "default": False,
             "help": "Opt in only if the device exposes a writable Gain feature."},
            {"key": "gain", "label": "Gain value (camera units)", "type": "float", "default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1},
            {"key": "apply_acquisition_frame_rate", "label": "Set camera acquisition frame rate", "type": "bool", "default": False,
             "help": "Opt in only if the device exposes a writable AcquisitionFrameRate feature."},
            {"key": "acquisition_frame_rate_hz", "label": "Camera acquisition frame rate (Hz)", "type": "float", "default": 3.0, "min": 0.01, "max": 1000.0, "step": 0.1},
        ]

    def probe(self, connection: str, settings=None, **options) -> list[dict]:
        if connection != "GenTL":
            raise ValueError("The built-in Harvester adapter only supports the GenTL connection.")
        h = self._harvester(dict(settings or {}))
        try:
            devices = []
            for idx, info in enumerate(h.device_info_list):
                serial = _device_value(info, "serial_number", "serial")
                device_id = serial or _device_value(info, "id_", "id", "user_defined_name") or str(idx)
                vendor = _device_value(info, "vendor")
                model = _device_value(info, "model")
                label = " ".join(v for v in (vendor, model, serial) if v) or f"GenTL camera {idx + 1}"
                devices.append({"device_id": device_id, "label": label, "vendor": vendor,
                                "model": model, "serial": serial,
                                "tl_type": _device_value(info, "tl_type"), "device_index": idx})
            return devices
        finally:
            try:
                h.reset()
            except Exception:
                pass

    @staticmethod
    def _node(node_map, candidate_names: tuple[str, ...]):
        for name in candidate_names:
            try:
                node = getattr(node_map, name)
            except Exception:
                node = None
            if node is not None:
                return name, node
        return None, None

    @staticmethod
    def _node_flag(node, name: str, default: bool = True) -> bool:
        value = getattr(node, name, default)
        try:
            return bool(value() if callable(value) else value)
        except Exception:
            return False

    def configure(self, device_id=None, settings=None, connection="GenTL", **options) -> dict:
        """Apply only explicitly enabled generic GenICam features and read back values."""
        if connection != "GenTL":
            raise ValueError("The built-in Harvester adapter only supports the GenTL connection.")
        cfg = dict(settings or {})
        requested = []
        if cfg.get("apply_exposure_time"):
            requested.append(("exposure_time_us", ("ExposureTime", "ExposureTimeAbs"), float(cfg.get("exposure_time_us", 5000.0))))
        if cfg.get("apply_gain"):
            requested.append(("gain", ("Gain", "GainRaw"), float(cfg.get("gain", 0.0))))
        if cfg.get("apply_acquisition_frame_rate"):
            requested.append(("acquisition_frame_rate_hz", ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"), float(cfg.get("acquisition_frame_rate_hz", 3.0))))
        if not requested:
            return {"ok": True, "configured": False, "applied": {}, "note": "No optional GenICam feature writes selected; device settings were not changed."}

        h = self._harvester(cfg)
        ia = None
        try:
            infos = list(h.device_info_list)
            selected = None
            for idx, info in enumerate(infos):
                candidates = {_device_value(info, "serial_number", "serial"),
                              _device_value(info, "id_", "id"), str(idx)}
                if device_id is None or str(device_id) in candidates:
                    selected = (idx, info)
                    break
            if selected is None:
                raise ValueError(f"GenTL device {device_id!r} was not found. Run camera discovery again.")
            idx, info = selected
            try:
                ia = h.create(search_key=info)
            except (TypeError, AttributeError):
                ia = h.create(search_key=idx)
            node_map = ia.remote_device.node_map
            actions = []
            for setting_key, candidates, desired in requested:
                if not np.isfinite(desired):
                    raise ValueError(f"Requested {setting_key} must be finite.")
                node_name, node = self._node(node_map, candidates)
                if node is None or not self._node_flag(node, "is_implemented") or not self._node_flag(node, "is_available"):
                    raise RuntimeError(f"This camera does not expose a usable GenICam node for {setting_key} ({', '.join(candidates)}). Disable that setting or use a vendor adapter.")
                if not self._node_flag(node, "is_writable"):
                    raise RuntimeError(f"GenICam node {node_name} is not writable on this camera in its current state.")
                lo = getattr(node, "min", None)
                hi = getattr(node, "max", None)
                if lo is not None and desired < float(lo):
                    raise ValueError(f"{setting_key}={desired} is below the camera's {node_name} minimum {lo}.")
                if hi is not None and desired > float(hi):
                    raise ValueError(f"{setting_key}={desired} is above the camera's {node_name} maximum {hi}.")
                actions.append((setting_key, node_name, node, desired))

            applied = {}
            for setting_key, node_name, node, desired in actions:
                node.value = desired
                applied[setting_key] = {"genicam_node": node_name, "requested": desired, "read_back": float(node.value)}
            return {"ok": True, "device_id": device_id, "applied": applied,
                    "note": "Only enabled GenICam nodes were written. Values were read back from the camera."}
        finally:
            if ia is not None:
                try:
                    ia.destroy()
                except Exception:
                    pass
            try:
                h.reset()
            except Exception:
                pass

    def healthcheck(self, device_id=None, connection="GenTL", settings=None, **options):
        devices = self.probe(connection, settings=settings)
        found = next((d for d in devices if str(d["device_id"]) == str(device_id)), None) if device_id else None
        return {"ok": bool(devices), "device_count": len(devices), "selected_device_found": found is not None if device_id else None,
                "note": "Discovery only; no radiometric calibration or acquisition has been verified."}

    def acquire_sequence(self, connection: str, device_id=None, duration_s=30.0,
                         frame_rate_hz=3.0, settings=None, **options) -> dict:
        if connection != "GenTL":
            raise ValueError("The built-in Harvester adapter only supports the GenTL connection.")
        cfg = dict(settings or {})
        h = self._harvester(cfg)
        ia = None
        frames: list[np.ndarray] = []
        device_timestamps_ns: list[float | None] = []
        host_receive_times: list[float] = []
        formats: list[str] = []
        start = None  # start the capture clock only after device setup succeeds
        target_frames = max(2, int(round(float(duration_s) * float(frame_rate_hz))))
        timeout = float(cfg.get("frame_timeout_s", 2.0))
        component_index = int(cfg.get("component_index", 0))
        selected = None
        try:
            infos = list(h.device_info_list)
            if not infos:
                raise RuntimeError("No GenTL cameras were discovered. Check the CTI producer, cable and camera driver.")
            for idx, info in enumerate(infos):
                candidates = {_device_value(info, "serial_number", "serial"),
                              _device_value(info, "id_", "id"), str(idx)}
                if device_id is None or str(device_id) in candidates:
                    selected = (idx, info)
                    break
            if selected is None:
                raise ValueError(f"GenTL device {device_id!r} was not found. Run camera discovery again.")
            idx, info = selected
            # Current Harvester API supports create(search_key=DeviceInfo or index).
            try:
                ia = h.create(search_key=info)
            except (TypeError, AttributeError):
                ia = h.create(search_key=idx)
            ia.start()
            start = time.monotonic()
            deadline = start + float(duration_s)
            while len(frames) < target_frames and time.monotonic() < deadline:
                try:
                    with ia.fetch(timeout=timeout) as buffer:
                        host_receive_times.append(time.monotonic())
                        components = list(getattr(buffer.payload, "components", []) or [])
                        if component_index >= len(components):
                            raise ValueError(
                                f"Requested image component {component_index}, but this payload has {len(components)} component(s)."
                            )
                        component = components[component_index]
                        raw = np.asarray(component.data)
                        height = int(getattr(component, "height", 0) or 0)
                        width = int(getattr(component, "width", 0) or 0)
                        if raw.ndim == 1 and height > 0 and width > 0 and raw.size == height * width:
                            raw = raw.reshape(height, width)
                        if raw.ndim != 2:
                            raise ValueError(f"GenTL component is not a scalar 2-D image (shape={raw.shape}). Select a suitable image component/pixel format.")
                        if raw.dtype.kind not in "uif":
                            raise ValueError(f"Unsupported GenTL component dtype: {raw.dtype}")
                        frames.append(raw.astype(np.float32, copy=True))
                        fmt = str(getattr(component, "data_format", "unknown"))
                        formats.append(fmt)
                        stamp_ns = getattr(buffer, "timestamp_ns", None)
                        try:
                            device_timestamps_ns.append(float(stamp_ns) if stamp_ns is not None else None)
                        except (TypeError, ValueError, OverflowError):
                            device_timestamps_ns.append(None)
                except Exception as exc:
                    # Harvester / GenTL producers may expose their timeout type
                    # through genicam rather than Python's built-in TimeoutError.
                    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.casefold():
                        continue
                    raise
            if len(frames) < 2:
                raise RuntimeError("Fewer than two frames were received before the capture ended.")
            if len(device_timestamps_ns) == len(frames) and all(v is not None for v in device_timestamps_ns):
                t = np.asarray(device_timestamps_ns, dtype=float) * 1e-9
                timestamp_source = "GenTL buffer timestamp_ns"
            else:
                # Never mix camera-clock and host-clock values in one timestamp vector.
                t = np.asarray(host_receive_times, dtype=float)
                timestamp_source = "host monotonic buffer-receive time (device timestamp unavailable for every frame)"
            # GenTL timestamp clocks may restart, wrap, or use model-specific ticks.
            # Reject non-monotonic device clocks rather than silently sorting frames.
            if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
                raise ValueError("GenTL/host timestamps were not strictly increasing. Check the producer's timestamp clock or use a vendor adapter with documented clock conversion.")
            device = info
            return {"frames": np.stack(frames), "timestamps_s": t - t[0],
                    "metadata": {"vendor": _device_value(device, "vendor"),
                                 "model": _device_value(device, "model"),
                                 "serial": _device_value(device, "serial_number", "serial"),
                                 "transport": "GenTL", "pixel_format": formats[0] if formats else "unknown",
                                 "pixel_formats_observed": sorted(set(formats)),
                                 "units": "camera_counts", "calibration_status": "not_decoded_by_generic_gentl_adapter",
                                 "radiometric_warning": "Generic GenTL returns the selected scalar pixel component. Camera counts are not Celsius; apply a vendor calibration decoder before selecting calibrated temperature units.",
                                 "actual_frame_count": len(frames), "requested_frame_rate_hz": float(frame_rate_hz),
                                 "actual_elapsed_s": float(t[-1] - t[0]),
                                 "timestamp_source": timestamp_source}}
        finally:
            if ia is not None:
                try:
                    if ia.is_acquiring():
                        ia.stop()
                except Exception:
                    pass
                try:
                    ia.destroy()
                except Exception:
                    pass
            try:
                h.reset()
            except Exception:
                pass


class HttpRadiometricMatrixAdapter:
    """User-configured HTTP JSON matrix bridge for compatible network cameras.

    The user supplies a vendor-supported endpoint or a small local gateway that
    returns scalar frame arrays. This adapter does not guess ISAPI/REST paths and
    never treats a palette RTSP/JPEG stream as radiometric temperature data.
    """

    def get_parameter_schema(self, connection=None):
        # Recorded JSON files need only decode/units controls; they do not need
        # network credentials or an endpoint URL.
        file_fields = [
            {"key": "frames_key", "label": "JSON frame-array field", "type": "str", "default": "frames"},
            {"key": "timestamps_key", "label": "JSON timestamps field", "type": "str", "default": "timestamps_s"},
            {"key": "units", "label": "Declared scalar units", "type": "select", "options": ["camera_counts", "relative_intensity", "°C"], "default": "camera_counts"},
            {"key": "calibration_confirmed", "label": "I have verified the endpoint returns calibrated °C values", "type": "bool", "default": False,
             "help": "This is an operator declaration stored in metadata, not a calibration performed by this adapter."},
        ]
        if connection == "Recorded file":
            return file_fields
        return [
            {"key": "endpoint_url", "label": "Radiometric matrix endpoint URL", "type": "str", "default": "",
             "help": "Enter the exact vendor-documented endpoint or your gateway URL. Must return JSON with a 3-D frames array."},
            {"key": "auth_mode", "label": "Authentication", "type": "select", "options": ["None", "HTTP Basic", "Bearer token"], "default": "None"},
            {"key": "username", "label": "Username", "type": "str", "default": ""},
            {"key": "password", "label": "Password", "type": "password", "default": ""},
            {"key": "bearer_token", "label": "Bearer token", "type": "password", "default": ""},
            *file_fields,
            {"key": "timeout_s", "label": "HTTP timeout (s)", "type": "float", "default": 20.0, "min": 1.0, "max": 180.0, "step": 1.0},
        ]

    @staticmethod
    def _url(settings: dict) -> str:
        url = str(settings.get("endpoint_url", "")).strip()
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Enter a valid http:// or https:// radiometric matrix API URL.")
        if parsed.username or parsed.password:
            raise ValueError("Do not embed credentials in the URL. Use the authentication fields.")
        return url

    @staticmethod
    def _request(settings: dict, query: dict | None = None) -> bytes:
        url = HttpRadiometricMatrixAdapter._url(settings)
        if query:
            parsed = urllib.parse.urlsplit(url)
            q = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
            q.extend((str(k), str(v)) for k, v in query.items())
            url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path,
                                           urllib.parse.urlencode(q), parsed.fragment))
        headers = {"Accept": "application/json", "User-Agent": "ThermoProp-Lab/1.5"}
        mode = str(settings.get("auth_mode", "None"))
        if mode == "HTTP Basic":
            token = base64.b64encode((str(settings.get("username", "")) + ":" + str(settings.get("password", ""))).encode()).decode()
            headers["Authorization"] = "Basic " + token
        elif mode == "Bearer token":
            token = str(settings.get("bearer_token", "")).strip()
            if not token:
                raise ValueError("Bearer token is required when Bearer token authentication is selected.")
            headers["Authorization"] = "Bearer " + token
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=float(settings.get("timeout_s", 20.0))) as response:
                content_type = response.headers.get("Content-Type", "")
                payload = response.read(1_500_000_001)
                if len(payload) > 1_500_000_000:
                    raise ValueError("HTTP matrix response exceeds the 1.5 GB upload safety limit.")
                if "json" not in content_type.lower() and not payload.lstrip().startswith((b"{", b"[")):
                    raise ValueError(f"Endpoint did not return JSON (Content-Type={content_type!r}). Configure a JSON matrix gateway or use scalar file import.")
                return payload
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Radiometric matrix API returned HTTP {exc.code}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach radiometric matrix API: {exc.reason}") from exc

    @staticmethod
    def _decode(payload: bytes, filename: str, settings: dict) -> dict:
        try:
            data = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise ValueError("Radiometric API payload is not valid UTF-8 JSON.") from exc
        if not isinstance(data, dict):
            raise ValueError("Radiometric API must return a JSON object with a frames array.")
        frames_key = str(settings.get("frames_key", "frames"))
        timestamps_key = str(settings.get("timestamps_key", "timestamps_s"))
        if frames_key not in data:
            raise ValueError(f"JSON field {frames_key!r} was not found. Available fields: {list(data)[:30]}")
        frames = np.asarray(data[frames_key])
        if frames.ndim not in (2, 3) or frames.dtype.kind not in "uif":
            raise ValueError("Radiometric matrix field must contain a numeric 2-D frame or 3-D frame sequence.")
        units = str(settings.get("units", "camera_counts"))
        calibration_confirmed = bool(settings.get("calibration_confirmed", False))
        if units == "°C" and not calibration_confirmed:
            raise ValueError("You selected °C but have not confirmed that the endpoint returns calibrated Celsius values. Verify the camera documentation/calibration first.")
        metadata = dict(data.get("metadata", {})) if isinstance(data.get("metadata", {}), dict) else {}
        metadata.update({"transport": "HTTP radiometric matrix API", "source_filename": filename,
                         "units": units, "calibration_status": "operator_confirmed" if units == "°C" and calibration_confirmed else "not_verified_by_adapter",
                         "radiometric_warning": "The adapter transports the vendor/gateway matrix without converting it. Verify endpoint units, calibration, emissivity and temperature scaling independently."})
        timestamps = data.get(timestamps_key)
        return {"frames": frames, "timestamps_s": timestamps, "metadata": metadata}

    def load_recording(self, payload: bytes, filename: str, settings=None, **options) -> dict:
        return self._decode(payload, filename, dict(settings or {}))

    def acquire_sequence(self, connection: str, device_id=None, duration_s=30.0,
                         frame_rate_hz=3.0, settings=None, **options) -> dict:
        if connection != "HTTP radiometric matrix API":
            raise ValueError("Choose the HTTP radiometric matrix API connection.")
        cfg = dict(settings or {})
        query = {"duration_s": float(duration_s), "frame_rate_hz": float(frame_rate_hz)}
        payload = self._request(cfg, query=query)
        result = self._decode(payload, "live_http_response.json", cfg)
        result["metadata"].update({"requested_duration_s": float(duration_s),
                                   "requested_frame_rate_hz": float(frame_rate_hz),
                                   "device_id": device_id,
                                   "capture_mode": "endpoint_returned_finite_sequence"})
        return result


def get_builtin_adapter(profile_id: str):
    """Return a built-in adapter only when its optional runtime is available."""
    if profile_id == "gentl":
        try:
            import harvesters.core  # noqa: F401
        except Exception:
            return None
        return GenTLAdapter()
    if profile_id == "http_matrix":
        return HttpRadiometricMatrixAdapter()
    return None
