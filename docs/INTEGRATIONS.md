# Camera, sensor, AI and research integrations — v1.5.0

This document distinguishes **selectable profiles**, **implemented application interfaces**, and **vendor SDK capabilities**. Selecting a vendor name does not connect hardware or install its SDK. The project intentionally does not bundle proprietary SDK binaries, licenses, firmware, or a universal hardware driver.

## 1. Camera profiles and what selecting one means

The application offers a broad selectable catalogue: Generic/research data; FLIR; InfraTec; Optris; Seek Thermal; HIKMICRO; Hikvision; Testo; Fluke; InfiRay; Guide Sensmart; Workswell; Xenics/Exosens; Telops; Sierra-Olympic; Axis; MOBOTIX; Basler; Allied Vision; IDS; LUCID; Teledyne DALSA; XIMEA; Raptor Photonics; New Imaging Technologies; smartphone thermal attachments/OEM cores; thermopile arrays; GenICam/GenTL; generic HTTP matrix; and Other/custom.

These are broad profiles, not promises of universal native-file or live-camera compatibility. Standard scalar imports remain the universal fallback when the camera can export numeric frames.

All profiles can use the standard scalar import formats supported by the application (`.npy`, `.npz`, `.tif`, `.tiff`, `.csv`, `.zip`). Native proprietary formats are offered in the uploader only when an installed adapter advertises a native decoder. Palette / pseudocolour exports are not accepted as calibrated temperature matrices.

The camera panel reads the selected adapter's capabilities and exposes:

- Device discovery (`probe`)
- Optional camera health check (`healthcheck`)
- Optional SDK parameter schema (`get_parameter_schema`)
- SDK setting application (`configure`)
- Finite single-camera capture (`acquire_sequence`)
- Finite multi-camera capture (`acquire_multi_sequence`)
- Native recording decoder (`load_recording`)

The per-camera parameter UI is generated from the installed adapter's schema. Supported field types are `bool`, `int`, `float`, `str`, `password`, and `select`; min/max/default/step fields are validated before being displayed. Password values are masked and redacted from analysis metadata. General acquisition metadata such as emissivity, reflected apparent temperature, transmissivity, humidity, camera distance, calibration ID, requested frame rate and spatial pitch remain user-configurable in the app. Those values are provenance fields; they are not silently applied to already-decoded radiometric arrays.

### Vendor notes — check the exact camera model

| Vendor / family | Official resource / interface | Integration boundary |
|---|---|---|
| FLIR Science IR | [Science Camera SDK](https://flir.custhelp.com/app/answers/detail/a_id/6623/~/science-camera-sdk---download) and [Science File SDK](https://flir.custhelp.com/app/answers/detail/a_id/6622/related/1) | These are distinct products. The Science Camera SDK supports camera control/acquisition; the Science File SDK reads stored radiometric files and does not control cameras. FLIR publishes a camera/SDK chooser because the right SDK varies by model. The Science Camera SDK requires a license. |
| InfraTec | [IRBGRAB / IRBACS SDK](https://www.infratec.eu/thermography/thermographic-software/sdk/) | IRBGRAB is the online/live SDK; IRBACS is the offline IRB reader/converter. InfraTec documents per-camera licensing for IRBGRAB and corrected-temperature export through IRBACS. |
| Optris PI / Xi | [OTC SDK](https://optris.com/software/otc-sdk/) | Official Python API, USB/Ethernet, temperature data, timestamps/calibration metadata and multiple cameras are documented. The adapter still needs to map the exact SDK result into the thermoprop scalar-frame contract. |
| Seek Thermal | [Official Python bindings](https://github.com/seekthermal/seekcamera-python) | Requires Seek SDK 4.x and a compatible physical OEM Core. The Python binding is not a universal driver for every consumer Seek device. |
| HIKMICRO | [Industrial camera product / integration capabilities](https://www.hikmicrotech.com/en/industrial-products/furnace-cameras/) | Certain industrial network cameras document 16/32-bit thermal matrices and SDK/ISAPI/RTSP outputs. Availability and payload semantics are model-specific. Decode actual thermal matrices, not palette video. |
| Testo | [Official Testo GitHub organization](https://github.com/orgs/testo/repositories) | Public model-specific API examples exist for certain Testo thermal cameras (for example, 87x families). Do not infer that all Testo models share one interface. |
| Fluke / InfiRay / Guide | Vendor model-specific API / SDK | Standard scalar import is available. Live/native support requires a confirmed model-specific API and a thermoprop adapter; no universal Python driver is assumed. |
| GenICam / GenTL | [Harvester](https://harvesters.readthedocs.io/) | Built-in generic acquisition is available with a vendor CTI producer. Pixel format, radiometric calibration and conversion from sensor counts remain camera-specific. A generic GenTL stream is not automatically temperature in °C. |
| Axis / MOBOTIX | [Axis](https://www.axis.com/) / [MOBOTIX](https://www.mobotix.com/) | Network profiles are metadata presets. Use a documented numeric matrix API or a tested vendor adapter; ordinary RTSP/palette video is not accepted as radiometric data. |
| Basler / Allied Vision / IDS / LUCID / Teledyne DALSA | Vendor SDKs plus [GenICam](https://www.emva.org/standards-technology/genicam/) | Prefer the generic GenTL profile when the exact model exposes a compatible CTI producer. Vendor-specific controls and calibration need adapters. |
| XIMEA / Raptor Photonics / NIT | Vendor SDK / numeric export | Scientific SWIR/MWIR/LWIR sensors may expose radiance/counts rather than object temperature; retain spectral and calibration metadata. |
| Smartphone cores / thermopile arrays | OEM SDK / I²C/SPI/serial adapter | Consumer attachments and low-resolution arrays are highly model-specific. Scalar data export works; live hardware needs an exact adapter. |

Vendor connection options are explicitly labelled model-specific wherever appropriate. They are not promises that every product from a vendor has a supported live SDK. An installed adapter must implement and report the required method.


## 1A. Broad camera catalogue and detector metadata

The camera selector includes profiles for Generic/research data, FLIR, InfraTec, Optris, Seek Thermal, HIKMICRO, Hikvision, Testo, Fluke, InfiRay, Guide Sensmart, Workswell, Xenics/Exosens, Telops, Sierra-Olympic, Axis, MOBOTIX, Basler, Allied Vision, IDS, LUCID, Teledyne DALSA, XIMEA, Raptor Photonics, New Imaging Technologies (NIT), smartphone thermal attachments/OEM cores, thermopile arrays, GenICam/GenTL, a generic HTTP matrix API and Other/custom instruments.

The separate **Camera / detector architecture** panel lets an operator record physical interface (USB/USB3 Vision, USB-C, GigE Vision/10GigE, Wi-Fi, HTTP/REST/ISAPI, RTSP/RTP, GenTL, Camera Link/Camera Link HS, CoaXPress, PCIe/frame grabber, serial, I²C/SPI and custom), form factor (handheld, fixed, scientific high-speed, industrial machine vision, network/IP, OEM core, mobile, UAV, thermopile array, multispectral/hyperspectral, line-scan and other), detector technology (microbolometer, cooled photon detector, thermopile, InSb, MCT/HgCdTe, InGaAs, QWIP and others), spectral band (NIR/SWIR/MWIR/LWIR/VLWIR/multiband), output representation and calibration status. These are provenance fields only. They do not imply that the selected vendor SDK is installed or that the uploaded values have been calibrated.

No finite menu can enumerate every OEM or future model. Choose **Other / custom instrument** and use scalar import, a compatible GenTL producer, the generic HTTP matrix gateway, or a third-party plugin. A new brand profile can be added without changing the analysis pipeline.

### Built-in generic connection methods

- **GenICam / GenTL:** The built-in Harvester adapter supports device discovery and finite scalar-component acquisition after installing the optional vision extra and the exact manufacturer's GenTL Producer (`.cti`). It returns camera counts/scalar payloads, pixel-format metadata and timestamps; it does not translate counts into Celsius or interpret vendor-specific calibration chunks. When standard GenICam nodes are exposed, the user may opt in to set `ExposureTime`, `Gain` and `AcquisitionFrameRate`; the adapter checks node availability, writability and the camera's reported min/max, then reads back values. Unsupported nodes are rejected clearly. Install with `pip install -e ".[vision]"` and provide the `.cti` file. Harvester uses the GenTL interface; it is not a replacement for the manufacturer's producer or calibration SDK.
- **HTTP radiometric matrix API:** The built-in configurable connector can load a JSON matrix recording or request a finite JSON sequence from an endpoint/gateway the operator documents. The JSON must include a numeric `frames` field shaped `(frames, height, width)` (or one 2-D frame), optional `timestamps_s`, and optional `metadata`. The adapter does not guess Hikvision/HIKMICRO/Axis/MOBOTIX ISAPI paths, decode RTSP, or treat palette JPEG/MP4 as radiometric matrices. Use it only with a vendor-documented numeric endpoint or a trusted local bridge.

Example JSON format is provided at `examples/http_matrix_recording.json`. Use `camera_counts` or `relative_intensity` unless the endpoint's calibration is independently documented. Selecting °C requires an explicit operator confirmation and is recorded as operator-confirmed metadata, not an independent calibration verification.

The UI now distinguishes **file import** from **live/native SDK support**. A missing adapter does not block standard scalar file imports. It means only that native proprietary decoding and/or live capture cannot be invoked until a compatible adapter is installed.

See the detailed family-by-family matrix in [CAMERA_SUPPORT_MATRIX.md](CAMERA_SUPPORT_MATRIX.md).

## 2. Camera adapter contract

A separately installed Python package registers an entry point whose name matches a profile ID:

```toml
[project.entry-points."thermoprop.camera_adapters"]
flir = "my_flir_adapter:get_adapter"
```

`get_adapter()` returns an object implementing any subset of these methods:

```python
class Adapter:
    def get_parameter_schema(self, connection=None) -> list[dict]: ...
    def probe(self, connection: str, **options) -> list[dict]: ...
    def healthcheck(self, device_id=None, **options) -> dict: ...
    def configure(self, device_id=None, settings=None, **options) -> dict: ...
    def acquire_sequence(self, connection, device_id=None, duration_s=30,
                         frame_rate_hz=3, settings=None, **options) -> dict: ...
    def acquire_multi_sequence(self, connection, device_ids, duration_s=30,
                               frame_rate_hz=3, settings=None, **options) -> dict: ...
    def load_recording(self, payload: bytes, filename: str, **options) -> dict: ...
```

A single recording returns:

```python
{
    "frames": scalar_frames,       # (n_frames, height, width), numeric
    "timestamps_s": timestamps,   # one strictly increasing timestamp/frame, or None
    "metadata": {"units": "°C", "model": "...", "serial": "..."}
}
```

The application rejects RGB/pseudocolour arrays, malformed frame dimensions, timestamp-count mismatches, non-monotonic frame timestamps, and severely corrupted arrays. Generic scalar import remains available when no camera SDK adapter is installed.

Multi-camera adapters return:

```python
{
    "recordings": {"camera_serial_1": recording_1, "camera_serial_2": recording_2},
    "metadata": {"synchronization_method": "shared hardware trigger"}
}
```

Every camera must provide timestamps and the adapter must document `synchronization_method`. `synchronization_verified_by_adapter` is treated as false unless the adapter explicitly sets it true. The application does not assume separate camera clocks are synchronized. It currently analyses one selected camera sequence at a time; multi-camera recording is retained as separate per-camera recordings rather than fused into one image tensor. A multi-device capture currently uses one selected vendor adapter/profile; heterogeneous vendor cameras should be captured separately unless a custom orchestration adapter provides and validates a common timebase.

Download `src/thermoprop/integrations/camera_adapter_template.py.txt` from the app for a starter contract. Install the vendor SDK separately and comply with its license. Adapter development should be tested using the specific camera model, firmware, operating system, SDK version and radiometric settings.

## 3. Physiological reference signals

The application can compare thermal ROI features with user-selected reference streams:

- LSCI perfusion index
- Laser-Doppler flux
- PPG amplitude / envelope
- ECG
- GSR / EDA
- External probe temperature
- Ambient temperature logger
- Other timestamped reference signals

The modalities may be imported through CSV, XDF or LSL. Each channel has independent name, modality, units, preprocessing and clock offset. CSV import requires numeric time and value columns. The importer records invalid rows, duplicate timestamps, reordered rows, interval variability and large gaps. Correlation/lag outputs are descriptive and do not correct for serial dependence.

### XDF / LabRecorder

Install the optional sensor dependencies:

```powershell
pip install -e ".[sensors]"
```

Upload an `.xdf`, read its streams, select a numeric stream/channel, assign its modality and units, and add it to the selected thermal recording. `pyxdf` is called with XDF clock synchronization enabled by default. Timestamp de-jitter is a separate user option and defaults **off** so real irregular intervals and dropped-frame gaps are not silently regularized. Clock-reset handling is enabled. The original XDF should be retained as the authoritative source.

### Live LSL

With `pylsl` installed, discover active LSL outlets and capture a finite numeric channel. The UI uses the outlet's unique stream UID when available and falls back to source ID. LSL inlet clock correction is enabled by default and refreshed during long captures; the latest finite correction is applied per received chunk and the correction history is retained. If the thermal camera timestamps use a different clock, enter a separately measured camera-to-LSL offset. LSL clock correction alone does not synchronize an unrelated camera clock. The capture metadata also records interval variability, large gaps, reordered samples and the clock domain actually used.

LSL documentation: [time synchronization](https://labstreaminglayer.readthedocs.io/info/time_synchronization.html). PyXDF documentation/source: [pyxdf](https://pypi.org/project/pyxdf/). pylsl package / platform library notes: [pylsl](https://pypi.org/project/pylsl/).

### Interpretation

LSCI and laser-Doppler can provide independent perfusion-related reference signals when measured correctly. They have their own units, sampling limits and motion sensitivities. PPG, ECG, GSR/EDA, probe temperature and ambient logs are complementary signals, not interchangeable local microvascular ground truth. Cross-correlation and lag selection are exploratory; they do not establish causality or absolute perfusion calibration.

## 4. AI and modelling integrations

### Thermal-domain ROI suggestions

Install:

```powershell
pip install -e ".[ai]"
```

Upload trusted YOLO segmentation weights trained or validated on thermal imagery from a comparable camera/domain. The model input is a documented normalization of scalar pixel values; predictions are suggestions that must be reviewed in the polygon editor. Generic RGB weights are not assumed to be valid for thermal imagery. A single-frame mask does not by itself perform temporal anatomical tracking.

### Supervised thermal-to-reference regression

Install scikit-learn and optionally XGBoost:

```powershell
pip install -e ".[ml-xgboost]"
```

The supervised model studio accepts participant/session/ROI summary rows paired with independent reference labels. It uses participant-grouped cross-validation (GroupKFold) to avoid placing rows from one participant in both train and test folds. At least six distinct participants and twelve rows are required. Frame-level rows are prohibited because they create pseudoreplication and data leakage. Outputs are predictions of the supplied reference target, not direct thermal blood-flow measurements. External cohort/site validation remains required.

### SHAP explainability

Install:

```powershell
pip install -e ".[explainability]"
```

Global SHAP explanations are exposed for Random Forest, Extra Trees and XGBoost regression pipelines. They explain how a fitted model uses the supplied cohort features; they are not causal effects and do not replace external validation.

### Experiment tracking / data versioning

- `pip install -e ".[tracking]"` enables optional local MLflow experiment logging.
- `pip install -e ".[data-versioning]"` enables optional DVC dataset/pipeline versioning.
- `pip install -e ".[wavelets]"` enables the optional continuous wavelet tools.

## 5. Scientific boundaries

- Surface thermography measures emitted IR radiation; temperature requires appropriate radiometric calibration and correction.
- A camera profile selection or metadata field does not calibrate raw counts.
- The apparent thermal-front direction/speed is not vascular direction or blood-flow velocity.
- The Pennes module is a forward sensitivity sandbox, not an inverse solver for an individual's perfusion.
- The optional paper-derived heat-balance index is a model output under its paper assumptions, not a direct blood-flow map.
- A machine-learning output is only as valid as its independent reference labels, cohort design, acquisition calibration and external validation.

The software is research-only and is not a medical device or a diagnostic system.

## Acquisition metadata versus SDK configuration

The general acquisition panel lets the operator record ambient temperature, relative humidity, acclimatization duration, camera distance, emissivity, reflected apparent temperature, transmissivity, window temperature, camera model/serial and calibration reference. These values are provenance/analysis metadata; they do not retroactively calibrate an uploaded array or reconfigure a physical camera. The selected vendor adapter may expose a separate dynamic SDK parameter schema for controls such as integration time, range, focus or frame rate. Only those adapter-declared controls are passed to the SDK.
