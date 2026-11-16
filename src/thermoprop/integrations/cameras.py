"""Capability-aware thermal-camera profiles and vendor plugin discovery.

This module intentionally does not claim that one SDK can control every camera.
Vendor SDKs are often licensed, OS-specific, or model-specific. A profile can
still be selected for metadata and standard exports even if its live SDK is not
installed. Native vendor files are decoded only by a matching installed plugin.

Plugin contract (Python entry point group ``thermoprop.camera_adapters``):

    [project.entry-points."thermoprop.camera_adapters"]
    flir = "my_flir_adapter:get_adapter"

The factory returns an object with optional methods:
    probe(connection: str, **options) -> list[dict]
    load_recording(payload: bytes, filename: str, **options) -> dict

load_recording must return ``{"frames": ndarray, "timestamps_s": ndarray|None,
"metadata": dict}``. Frames must be scalar calibrated-temperature or raw
sensor-value arrays; pseudocolour RGB exports must not be decoded as temperature.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraProfile:
    id: str
    label: str
    native_extensions: tuple[str, ...]
    standard_extensions: tuple[str, ...]
    offline_sdk: str
    live_sdk: str
    live_connections: tuple[str, ...]
    calibration_notes: str
    documentation_url: str
    status_note: str


PROFILES: tuple[CameraProfile, ...] = (
    CameraProfile(
        id="generic", label="Generic / research dataset", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Not required for supported scalar exports.",
        live_sdk="No universal thermal-camera live driver. Use a supported GenTL producer, a configured HTTP radiometric-matrix gateway, or a vendor adapter.",
        live_connections=(),
        calibration_notes="Enter the actual units and calibration provenance. A unit selector does not calibrate values.",
        documentation_url="",
        status_note="Standard scalar sequence import is available. This is the most portable option across camera manufacturers.",
    ),
    CameraProfile(
        id="flir", label="FLIR / Teledyne FLIR (Science, A-series, Boson, Tau, Lepton)", native_extensions=(".seq", ".ats", ".sfmov", ".fff"),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Science File SDK supports native scientific formats such as SEQ/ATS/SFMOV. flirpy is a community library for selected Lepton/Boson/Tau cores; model/firmware support varies.",
        live_sdk="Science Camera SDK for supported science cameras; Spinnaker for supported machine-vision models; mobile and Atlas SDKs cover different product lines. SDK selection is model-specific.",
        live_connections=("USB", "Ethernet", "Serial / core interface (supported cores)"),
        calibration_notes="Preserve native radiometric data, per-frame metadata, detector temperature and calibration parameters. Do not turn display palette RGB into temperature.",
        documentation_url="https://flir.custhelp.com/app/answers/detail/a_id/3000/~/flir-cameras---which-sdk-should-i-use-for-my-camera-application%253F",
        status_note="FLIR uses multiple SDK families. Install the SDK matching the exact camera family and a ThermoProp plugin; the Science File SDK is for file decoding, not camera control.",
    ),
    CameraProfile(
        id="infratec", label="InfraTec (ImageIR / VarioCAM / IRB)", native_extensions=(".irb",),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="IRBACS-SDK reads stored IRB recordings and metadata and can convert digital values to corrected temperature data.",
        live_sdk="IRBGRAB-SDK provides online acquisition and parameterization; suitable camera add-on and per-camera licensing apply.",
        live_connections=("USB / IRBGRAB-SDK", "Ethernet / IRBGRAB-SDK"),
        calibration_notes="Prefer corrected temperature output; retain camera model, detector, calibration, emissivity and per-frame timing metadata.",
        documentation_url="https://www.infratec-infrared.com/thermography/thermographic-software/sdk/",
        status_note="InfraTec separates IRBACS offline processing from IRBGRAB online acquisition. Each needs its matching vendor SDK and adapter.",
    ),
    CameraProfile(
        id="optris", label="Optris (PI / Xi series)", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Use scalar temperature exports or Optris SDK workflows. The official OTC SDK exposes thermal data and conversion functions.",
        live_sdk="OTC SDK provides Python 3 bindings, thermal streaming, temperature conversion, setup/control and USB/Ethernet interfaces for supported PI/Xi models.",
        live_connections=("USB / OTC SDK", "Ethernet / OTC SDK"),
        calibration_notes="Use SDK temperature data where available; record temperature range, emissivity, transmissivity, distance, calibration and timestamps.",
        documentation_url="https://optris.github.io/otcsdk_downloads/",
        status_note="The official Optris OTC SDK has Python bindings. Install its native runtime/SDK and an Optris ThermoProp adapter; Python/NumPy ABI compatibility matters.",
    ),
    CameraProfile(
        id="seek", label="Seek Thermal (OEM Core / Mosaic / Micro Core)", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Export scalar frame values from a compatible Seek SDK workflow.",
        live_sdk="Official seekcamera-python bindings require Seek Thermal SDK 4.x and a physically connected compatible OEM core; consumer products are not all exposed through this SDK.",
        live_connections=("USB / Seek OEM SDK",),
        calibration_notes="Confirm whether output values are calibrated temperature, radiometric counts or display-scaled values. Store telemetry and FFC state if available.",
        documentation_url="https://github.com/seekthermal/seekcamera-python",
        status_note="Official Python bindings exist for supported OEM cores. Install Seek SDK runtime and a matching adapter; compatibility is not universal across Seek products.",
    ),
    CameraProfile(
        id="hikmicro", label="HIKMICRO", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="No universal native decoder is assumed. Use a model-supported radiometric export from HIKMICRO Studio/Analyzer or a vendor-approved decoder.",
        live_sdk="Model-specific SDK and network integration. Some industrial models document 16/32-bit full thermographic matrices over RTSP, ISAPI and SDK; handheld models differ.",
        live_connections=("Ethernet / vendor SDK (supported models)",),
        calibration_notes="Use the full thermographic matrix, not the YUV/palette video stream. Verify pixel encoding, data length, temperature mapping, emissivity and timestamp behavior per model.",
        documentation_url="https://www.hikmicrotech.com/",
        status_note="Some industrial models expose numeric thermographic matrices; other models expose only image/video exports. Use the generic HTTP matrix connector only with a vendor-documented numeric endpoint.",
    ),
    CameraProfile(
        id="hikvision", label="Hikvision thermographic / radiometric network cameras", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Use the model-supported temperature-data export or a vendor SDK/API decoder.",
        live_sdk="Some thermographic network models offer SDK/ISAPI/RTSP temperature data interfaces. Endpoint, pixel data length and temperature mapping are firmware/model-specific.",
        live_connections=("Ethernet / SDK or ISAPI (supported models)", "RTSP thermographic matrix (supported models)"),
        calibration_notes="Confirm that the endpoint returns numeric temperature matrices, not a colorized video stream. Keep the data-length and temperature-mapping-table metadata.",
        documentation_url="https://www.hikvision.com/",
        status_note="Network integration is model/firmware-specific. Configure the generic HTTP matrix connector or install a vendor-specific adapter; this profile does not guess ISAPI URLs.",
    ),
    CameraProfile(
        id="testo", label="Testo thermal cameras", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Use model-supported radiometric export or vendor software conversion to scalar matrices. No universal native Python decoder is assumed.",
        live_sdk="Live API/SDK access depends on the exact camera series, firmware and software licensing; verify with Testo for the selected model.",
        live_connections=("USB / vendor interface (model-specific)", "Ethernet / vendor interface (model-specific)"),
        calibration_notes="Retain emissivity, reflected temperature, distance, humidity, calibration and original radiometric file metadata.",
        documentation_url="https://www.testo.com/",
        status_note="Scalar exports work. Native decoding and live control require a confirmed model-specific adapter integration.",
    ),
    CameraProfile(
        id="fluke", label="Fluke thermal cameras", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Use a model-supported radiometric export or vendor-approved decoder. Proprietary file compatibility must be confirmed per model.",
        live_sdk="No universal live Python driver is assumed; check model-specific SDK/API and licensing with Fluke.",
        live_connections=("Vendor SDK / API (model-specific)",),
        calibration_notes="Preserve radiometric metadata and emissivity/reflected-temperature settings; do not analyse palette exports as temperatures.",
        documentation_url="https://www.fluke.com/",
        status_note="Scalar exports are supported; native decoding and live acquisition require a compatible model-specific adapter.",
    ),
    CameraProfile(
        id="infiray", label="InfiRay / Infisense", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Use a compatible radiometric export or vendor software conversion to scalar arrays.",
        live_sdk="SDK availability and interface depend on the camera core/product family and distribution agreement.",
        live_connections=("USB / vendor SDK (model-specific)", "Ethernet / vendor API (model-specific)"),
        calibration_notes="Determine whether data represent Celsius, raw detector counts or display intensity; retain calibration and detector metadata.",
        documentation_url="https://www.infiray.com/",
        status_note="Vendor-specific driver and file decoding need a matching adapter; the profile does not assert universal InfiRay SDK compatibility.",
    ),
    CameraProfile(
        id="guide", label="Guide Sensmart", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Use scalar/radiometric exports supported by the selected Guide model and software.",
        live_sdk="Confirm model-specific SDK/API and licensing with Guide Sensmart.",
        live_connections=("USB / vendor SDK (model-specific)", "Ethernet / vendor API (model-specific)"),
        calibration_notes="Keep original radiometric file and all temperature mapping/calibration metadata.",
        documentation_url="https://www.guideir.com/",
        status_note="Scalar exports are portable. Native and live workflows require a matching vendor integration.",
    ),
    CameraProfile(
        id="workswell", label="Workswell (WIRIS / thermoIMAGER integrations)", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Use model-specific radiometric exports / SDK conversion. Supported payloads vary by camera and integration package.",
        live_sdk="Check exact model, interface and SDK availability with Workswell; do not assume standard RTSP carries radiometric pixels.",
        live_connections=("USB / vendor interface (model-specific)", "Ethernet / vendor interface (model-specific)"),
        calibration_notes="Retain temperature calibration, camera synchronization, lens and environmental metadata.",
        documentation_url="https://www.workswell-thermal-camera.com/",
        status_note="Vendor adapter required for native decoding or live acquisition; scalar temperature exports remain supported.",
    ),
    CameraProfile(
        id="xenics", label="Xenics / Exosens infrared cameras", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Use the model-supported Xeneth/SDK export or a documented numeric data stream.",
        live_sdk="Xeneth SDK / camera interface support depends on model, detector and transport interface.",
        live_connections=("USB / Xeneth SDK (supported models)", "GigE / Xeneth or GenTL (supported models)"),
        calibration_notes="Many scientific IR sensors provide radiance/count data rather than direct object temperature; apply a validated calibration chain before labelling °C.",
        documentation_url="https://www.exosens.com/",
        status_note="A vendor SDK or compatible GenTL producer is required; spectral band and sensor response must be documented.",
    ),
    CameraProfile(
        id="telops", label="Telops scientific infrared cameras", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Use Telops software/SDK-supported scientific data exports; native format support is system-specific.",
        live_sdk="Model-specific Telops acquisition software/API and interface package required.",
        live_connections=("GigE / vendor SDK (model-specific)", "Camera Link / frame grabber (model-specific)"),
        calibration_notes="Retain detector calibration, spectral response, integration time, non-uniformity correction and frame timing.",
        documentation_url="https://www.telops.com/",
        status_note="Scientific high-speed/spectral systems need a vendor/frame-grabber adapter. Do not treat radiance as surface Celsius without a conversion model.",
    ),
    CameraProfile(
        id="sierra_olympic", label="Sierra-Olympic / infrared OEM", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Use the OEM module's documented scalar or radiometric export.",
        live_sdk="SDK and protocol depend on OEM core, camera module and integration partner.",
        live_connections=("USB / OEM SDK (model-specific)", "Ethernet / OEM SDK (model-specific)"),
        calibration_notes="Record the OEM core, firmware, lens, FPA data, calibration and output units.",
        documentation_url="https://www.sierraolympic.com/",
        status_note="OEM module-specific integration required. Use the custom adapter contract for modules not covered by a public generic standard.",
    ),
    CameraProfile(
        id="axis", label="Axis thermal / radiometric network cameras", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Use model-supported radiometric exports or a documented numeric temperature API. Ordinary network video may be palette imagery only.",
        live_sdk="Network APIs and thermal metadata vary by model and firmware. Use a documented numeric matrix endpoint or a tested Axis-specific adapter.",
        live_connections=("Ethernet / documented API adapter (model-specific)",),
        calibration_notes="Do not treat ordinary RTSP video or colourized thermal streams as calibrated temperature matrices.",
        documentation_url="https://www.axis.com/",
        status_note="Profile for network thermal cameras. Native radiometric matrix access is model/firmware-specific; use the generic HTTP matrix connector only with a documented numeric endpoint.",
    ),
    CameraProfile(
        id="mobotix", label="MOBOTIX thermal / dual-thermal systems", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Use model-supported radiometric data exports or a documented API/gateway that returns numeric matrices.",
        live_sdk="Thermal analytics/API capabilities differ across MOBOTIX product lines and firmware; a model-specific adapter is required for live radiometric acquisition.",
        live_connections=("Ethernet / vendor API adapter (model-specific)",),
        calibration_notes="Verify whether the selected camera exposes numeric radiometric data or only thermal video/alarms.",
        documentation_url="https://www.mobotix.com/",
        status_note="Vendor and firmware dependent. Standard scalar exports work; live radiometric support requires a model-specific adapter or documented matrix gateway.",
    ),
    CameraProfile(
        id="basler", label="Basler infrared / machine-vision cameras", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Use scalar exports or vendor-supported pylon/GenTL acquisition and decoding.",
        live_sdk="pylon or a compatible GenTL Producer can provide acquisition for supported machine-vision models; sensor output may be counts/radiance, not temperature.",
        live_connections=("USB / pylon adapter (model-specific)", "GigE / pylon adapter (model-specific)"),
        calibration_notes="Check detector type, pixel format, spectral response and radiometric calibration before assigning temperature units.",
        documentation_url="https://www.baslerweb.com/",
        status_note="Use the GenICam / GenTL profile for built-in generic Harvester acquisition where the model exposes a compatible CTI producer; pylon-specific features require a vendor adapter.",
    ),
    CameraProfile(
        id="allied_vision", label="Allied Vision infrared cameras", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Use scalar export or Vimba/GenTL-supported acquisition data.",
        live_sdk="Vimba X and compatible GenTL producers support selected cameras; integration depends on model and interface.",
        live_connections=("USB3 / Vimba adapter (model-specific)", "GigE / Vimba or GenTL adapter"),
        calibration_notes="Camera output commonly requires a detector-specific calibration path before it represents surface temperature.",
        documentation_url="https://www.alliedvision.com/",
        status_note="Use the GenICam / GenTL profile for generic scalar capture if a compatible CTI producer is available; Vimba-specific controls need an adapter.",
    ),
    CameraProfile(
        id="ids", label="IDS imaging / infrared cameras", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Use scalar exports or IDS peak / uEye-supported data conversion.",
        live_sdk="IDS peak, uEye or GenTL access depends on camera family, transport and firmware.",
        live_connections=("USB / IDS adapter (model-specific)", "GigE / IDS or GenTL adapter"),
        calibration_notes="Confirm the sensor's spectral band and whether payload values are calibrated temperature or detector counts.",
        documentation_url="https://www.ids-imaging.com/",
        status_note="Generic GenTL may be available for compatible devices; vendor-specific controls and calibration conversion need a matching adapter.",
    ),
    CameraProfile(
        id="lucid", label="LUCID Vision Labs infrared cameras", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Use scalar exports or Arena SDK / GenTL-supported data.",
        live_sdk="Arena SDK and GenTL Producer support selected Triton/Atlas models and interfaces; check the exact camera.",
        live_connections=("USB3 / Arena adapter (model-specific)", "GigE / Arena or GenTL adapter"),
        calibration_notes="Thermal detector output and calibration are model-specific; do not infer Celsius from Mono16 without a mapping.",
        documentation_url="https://thinklucid.com/",
        status_note="Use built-in GenTL capture when a compatible CTI producer exists; Arena-specific features require a vendor adapter.",
    ),
    CameraProfile(
        id="teledyne_dalsa", label="Teledyne DALSA infrared / scientific cameras", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Use Sapera LT / GenTL or documented scalar exports as supported by the camera.",
        live_sdk="Sapera LT, Camera Link frame grabbers and GenTL support vary by model and acquisition hardware.",
        live_connections=("GigE / GenTL adapter", "Camera Link / frame-grabber adapter"),
        calibration_notes="High-speed and scientific sensors may return radiance/counts; preserve integration time and detector calibration.",
        documentation_url="https://www.teledynedalsa.com/",
        status_note="GenTL works only for compatible CTI-supported payloads; Camera Link systems need a supported frame grabber and adapter.",
    ),
    CameraProfile(
        id="ximea", label="XIMEA infrared / hyperspectral cameras", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Use xiAPI / xiAPI.NET or vendor-supported scalar data export.",
        live_sdk="xiAPI and model-specific interfaces are required; a GenTL path is possible only when a matching producer is supplied.",
        live_connections=("USB / xiAPI adapter (model-specific)", "Ethernet / xiAPI or GenTL adapter"),
        calibration_notes="Distinguish spectral radiance, digital counts and calibrated temperature. Hyperspectral cubes require a wavelength-aware import path.",
        documentation_url="https://www.ximea.com/",
        status_note="The profile supports metadata and scalar export workflows; spectral-cube import and xiAPI live capture require dedicated adapters.",
    ),
    CameraProfile(
        id="raptor_photonics", label="Raptor Photonics / SWIR-MWIR-LWIR cameras", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Use the supported Raptor SDK export or a numeric acquisition export.",
        live_sdk="Raptor camera interfaces and SDKs depend on detector and camera model; scientific sensors may output calibrated radiance rather than object temperature.",
        live_connections=("CameraLink / frame-grabber adapter", "USB / vendor SDK adapter", "GigE / vendor SDK adapter"),
        calibration_notes="Record spectral band, detector material, integration time, non-uniformity correction and radiometric calibration chain.",
        documentation_url="https://www.raptorphotonics.com/",
        status_note="Scientific detector profile. Native and live acquisition require a model-specific driver; generic scalar export remains available.",
    ),
    CameraProfile(
        id="nit", label="New Imaging Technologies (NIT) infrared cameras", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Use model-supported SDK or scalar data export.",
        live_sdk="NIT camera SDK and transport support are model-specific; confirm the numeric data format with the vendor.",
        live_connections=("USB / vendor SDK adapter", "GigE / vendor SDK or GenTL adapter"),
        calibration_notes="Preserve detector output units and calibration coefficients; many SWIR products are not direct surface-temperature imagers.",
        documentation_url="https://www.n-imaging-technologies.com/",
        status_note="Profile covers infrared detector families; temperature conversion needs a validated radiometric model.",
    ),
    CameraProfile(
        id="smartphone_core", label="Smartphone thermal attachments / OEM mobile cores", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Export scalar temperature matrices from a compatible mobile/OEM application; JPEG screenshots are not radiometric.",
        live_sdk="Mobile SDK availability depends on the attachment (for example selected FLIR/Seek products), OS and model.",
        live_connections=("USB-C / mobile OEM adapter (model-specific)",),
        calibration_notes="Preserve mobile app export metadata and confirm whether per-pixel values are calibrated or display-scaled.",
        documentation_url="https://www.flir.com/",
        status_note="Smartphone accessories vary widely. Use a documented radiometric export or compatible OEM SDK adapter.",
    ),
    CameraProfile(
        id="thermal_array", label="Thermopile arrays / low-resolution thermal sensors", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Import scalar matrices from sensor libraries or microcontroller/DAQ exports.",
        live_sdk="Sensor-vendor I2C/SPI/USB/serial drivers are hardware-specific; a small adapter can return timestamped scalar arrays.",
        live_connections=("USB / serial / DAQ adapter (model-specific)",),
        calibration_notes="Record sensor geometry, per-pixel calibration, emissivity and ambient compensation; low-resolution arrays may not resolve anatomical ROIs.",
        documentation_url="",
        status_note="For thermopile arrays and low-resolution thermal modules. A sensor adapter is required for live capture; CSV/NPY scalar import works.",
    ),
    CameraProfile(
        id="gentl", label="GenICam / GenTL industrial thermal cameras (Basler, Allied Vision, IDS, LUCID, Teledyne DALSA, etc.)", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Harvester reads frames through an installed GenTL Producer (.cti). The camera vendor's producer, pixel format and radiometric calibration decoder may still be required.",
        live_sdk="Built-in Harvester adapter supports device discovery and finite scalar-component capture when `.[vision]` and a valid CTI producer are installed.",
        live_connections=("GenTL",),
        calibration_notes="Generic GenTL returns scalar component values and timestamps. Counts are not automatically Celsius; a vendor calibration decoder is needed for temperature conversion.",
        documentation_url="https://www.emva.org/standards-technology/genicam/",
        status_note="Built-in generic GenTL acquisition is available when Harvester and the manufacturer's CTI producer are installed. Numeric radiometric decoding remains camera-specific.",
    ),
    CameraProfile(
        id="http_matrix", label="Generic HTTP / ISAPI radiometric matrix API", native_extensions=(".json",),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="No vendor SDK required when a camera or local gateway provides a documented JSON matrix response.",
        live_sdk="Built-in HTTP adapter requests a finite scalar sequence from a user-configured endpoint. It does not guess vendor-specific ISAPI paths.",
        live_connections=("HTTP radiometric matrix API",),
        calibration_notes="Endpoint must return numeric frames, timestamps and metadata. Palette RTSP/JPEG is not supported as radiometric data. Celsius requires explicit operator confirmation and calibration provenance.",
        documentation_url="",
        status_note="Built-in configurable JSON matrix connector. Use the documentation for your own camera or gateway; this profile does not assume a vendor endpoint. Numeric matrices only, never false-colour video frames.",
    ),
    CameraProfile(
        id="other", label="Other vendor / OEM module / custom camera", native_extensions=(),
        standard_extensions=(".npy", ".npz", ".tif", ".tiff", ".csv", ".zip"),
        offline_sdk="Use a vendor export to scalar radiometric arrays or create a small adapter plugin.",
        live_sdk="Requires a vendor SDK, GenTL Producer, documented network matrix endpoint or custom adapter.",
        live_connections=("USB / custom adapter", "Ethernet / custom adapter", "Serial / custom adapter"),
        calibration_notes="Document sensor class, spectral band, detector output, calibration chain, emissivity and scaling. Unknown outputs remain camera counts.",
        documentation_url="",
        status_note="The plugin contract is vendor-neutral. Add a custom adapter without modifying the analysis pipeline.",
    ),
)


# Broad hardware descriptors are independent of vendor profiles and drivers.
# They document the instrument configuration; they are not compatibility claims.
CAMERA_FORM_FACTORS: tuple[str, ...] = (
    "Unknown / not specified", "Handheld radiometric imager", "Fixed-mount thermal camera",
    "Scientific high-speed infrared camera", "Industrial machine-vision camera",
    "Radiometric network / IP camera", "OEM thermal core / module", "Smartphone thermal attachment",
    "UAV / airborne thermal payload", "Thermopile array / low-resolution thermal sensor",
    "Multispectral / hyperspectral infrared system", "Thermal line-scan / scanning system",
    "Microscopy / close-up thermal system", "Other / custom instrument",
)
DETECTOR_TECHNOLOGIES: tuple[str, ...] = (
    "Unknown / vendor-specific", "Uncooled microbolometer", "Cooled photon detector",
    "Thermopile array", "InSb detector", "MCT / HgCdTe detector", "InGaAs detector",
    "QWIP detector", "Pyroelectric detector", "CMOS / CCD silicon sensor",
    "Hybrid / multi-detector system", "Other / custom detector",
)
CAMERA_INTERFACES: tuple[str, ...] = (
    "Unknown / not specified", "USB 2.0", "USB 3.x / USB3 Vision", "USB-C / mobile accessory",
    "GigE Vision / Ethernet", "10 GigE / high-speed Ethernet", "Wi-Fi / wireless bridge",
    "HTTP / REST / ISAPI numeric API", "RTSP / RTP video (visual stream only unless numeric matrix is documented)",
    "GenICam / GenTL CTI Producer", "Camera Link / frame grabber", "Camera Link HS",
    "CoaXPress / frame grabber", "PCIe / embedded frame grabber", "Thunderbolt / PCIe bridge",
    "Serial / UART", "I²C / SPI / embedded DAQ", "Custom / OEM interface",
)
SPECTRAL_BANDS: tuple[str, ...] = (
    "Unknown / not specified", "NIR", "SWIR", "MWIR", "LWIR", "VLWIR",
    "Multiband / hyperspectral", "Other",
)
DETECTOR_OUTPUT_TYPES: tuple[str, ...] = (
    "Unknown / not documented", "Calibrated radiometric temperature (°C)",
    "Calibrated radiometric temperature (°F)", "Radiance / calibrated signal",
    "Linearized detector counts", "Raw detector counts / digital levels",
    "Relative / normalized intensity", "False-colour RGB / palette image (visual only)",
)
CALIBRATION_STATES: tuple[str, ...] = (
    "Unknown / not verified", "Factory calibrated (certificate available)",
    "Vendor calibrated (certificate not recorded)", "User calibrated against reference",
    "Raw / uncalibrated detector data", "Relative intensity only",
)


def get_profiles() -> tuple[CameraProfile, ...]:
    return PROFILES


def get_profile(profile_id: str) -> CameraProfile:
    for profile in PROFILES:
        if profile.id == profile_id:
            return profile
    raise KeyError(f"Unknown camera profile: {profile_id}")


def discover_camera_adapters() -> dict[str, Any]:
    """Discover optional adapters without importing vendor SDKs at app startup."""
    discovered: dict[str, Any] = {}
    try:
        eps = metadata.entry_points()
        group = eps.select(group="thermoprop.camera_adapters") if hasattr(eps, "select") else eps.get("thermoprop.camera_adapters", [])
        for ep in group:
            discovered[ep.name.casefold()] = ep
    except Exception:
        return {}
    return discovered


def get_adapter(profile_id: str):
    """Load an explicit vendor plugin, or a built-in protocol adapter.

    An installed third-party entry point takes precedence so users can supply a
    tested vendor-specific implementation for a camera family.
    """
    entry = discover_camera_adapters().get(profile_id.casefold())
    if entry is not None:
        factory = entry.load()
        return factory() if callable(factory) else factory
    try:
        from .builtin_camera_adapters import get_builtin_adapter
        return get_builtin_adapter(profile_id)
    except Exception:
        # Optional acquisition dependencies must never block recorded-file analysis.
        return None


def _adapter_method_names(adapter: Any | None) -> set[str]:
    if adapter is None:
        return set()
    return {name for name in (
        "probe", "load_recording", "acquire_sequence", "configure",
        "get_parameter_schema", "healthcheck", "acquire_multi_sequence"
    ) if callable(getattr(adapter, name, None))}


def _validated_recording(result: Any, *, source: str) -> dict[str, Any]:
    """Validate the adapter contract before arrays enter the analysis pipeline."""
    if not isinstance(result, dict) or "frames" not in result:
        raise ValueError(f"{source} adapter must return a mapping containing 'frames'.")
    frames = np.asarray(result["frames"])
    if frames.ndim == 2:
        frames = frames[None, ...]
    if frames.ndim == 4 and frames.shape[-1] in (3, 4):
        raise ValueError("Camera adapter returned RGB/pseudocolour frames. Return scalar radiometric or documented scalar sensor values.")
    if frames.ndim != 3:
        raise ValueError(f"Camera adapter returned shape {frames.shape}; expected (frames, height, width).")
    if frames.shape[0] < 2 or min(frames.shape[1:]) < 8:
        raise ValueError("Camera adapter must return at least 2 frames with spatial dimensions of at least 8 × 8.")
    if frames.dtype.kind not in "uif":
        raise ValueError(f"Camera frames must be numeric; received {frames.dtype}.")
    if frames.nbytes > 1_500_000_000:
        raise ValueError("Camera sequence exceeds the 1.5 GB safety limit. Capture a shorter segment.")
    finite_fraction = float(np.isfinite(frames).mean())
    if finite_fraction < 0.5:
        raise ValueError(f"Only {finite_fraction:.1%} of camera pixels are finite; rejecting corrupted/invalid acquisition.")
    frames = frames.astype(np.float32, copy=False)
    timestamps = result.get("timestamps_s")
    if timestamps is not None:
        timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
        if len(timestamps) != len(frames):
            raise ValueError("Adapter timestamp count does not match the number of frames.")
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError("Adapter timestamps must be finite and strictly increasing. Do not silently sort camera frames.")
    metadata_value = result.get("metadata", {}) or {}
    if not isinstance(metadata_value, dict):
        raise ValueError("Adapter metadata must be a dictionary.")
    metadata_value = dict(metadata_value)
    metadata_value.setdefault("adapter_source", source)
    metadata_value.setdefault("finite_pixel_fraction", finite_fraction)
    return {"frames": frames, "timestamps_s": timestamps, "metadata": metadata_value}


def get_parameter_schema(profile_id: str, connection: str | None = None) -> list[dict[str, Any]]:
    """Return a validated, optional UI schema supplied by the selected camera adapter.

    Supported field types: bool, int, float, str and select. Adapters remain
    responsible for applying parameters through their SDK; the app only collects
    and passes them through.
    """
    adapter = get_adapter(profile_id)
    if adapter is None or not callable(getattr(adapter, "get_parameter_schema", None)):
        return []
    method = adapter.get_parameter_schema
    try:
        schema = method(connection=connection)
    except TypeError:
        schema = method()
    if schema is None:
        return []
    if not isinstance(schema, (list, tuple)) or len(schema) > 50:
        raise ValueError("Camera adapter parameter schema must be a list with at most 50 fields.")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    allowed_types = {"bool", "int", "float", "str", "password", "select"}
    for item in schema:
        if not isinstance(item, dict):
            raise ValueError("Each camera parameter schema item must be a dictionary.")
        key = str(item.get("key", "")).strip()
        kind = str(item.get("type", "str")).casefold()
        if not key or not key.replace("_", "").isalnum() or len(key) > 64 or key in seen:
            raise ValueError(f"Invalid or duplicate camera parameter key: {key!r}")
        if kind not in allowed_types:
            raise ValueError(f"Unsupported camera parameter type {kind!r} for {key!r}")
        field = dict(item)
        field["key"] = key
        field["type"] = kind
        field["label"] = str(item.get("label", key.replace("_", " ").title()))[:100]
        field["help"] = str(item.get("help", ""))[:300]
        if kind == "select":
            options = item.get("options")
            if not isinstance(options, (list, tuple)) or not options or len(options) > 100:
                raise ValueError(f"Select parameter {key!r} needs a non-empty options list.")
            field["options"] = [str(v) for v in options]
            if any(not value for value in field["options"]) or len(set(field["options"])) != len(field["options"]):
                raise ValueError(f"Select parameter {key!r} has empty or duplicate options.")
            if str(field.get("default", field["options"][0])) not in field["options"]:
                field["default"] = field["options"][0]
        if kind in {"int", "float"}:
            lo, hi = item.get("min"), item.get("max")
            cast = int if kind == "int" else float
            try:
                lo_value = float(lo) if lo is not None else None
                hi_value = float(hi) if hi is not None else None
                if ((lo_value is not None and not np.isfinite(lo_value))
                        or (hi_value is not None and not np.isfinite(hi_value))):
                    raise ValueError("bounds must be finite")
                if lo_value is not None and hi_value is not None and hi_value <= lo_value:
                    raise ValueError("maximum must be greater than minimum")
                fallback = lo if lo is not None else (0 if kind == "int" else 0.0)
                default_value = cast(item.get("default", fallback))
                step_value = cast(item.get("step", 1 if kind == "int" else 0.1))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid numeric bounds/default/step for camera parameter {key!r}: {exc}") from exc
            if kind == "float" and (not np.isfinite(default_value) or not np.isfinite(step_value)):
                raise ValueError(f"Non-finite default or step for camera parameter {key!r}")
            if step_value <= 0 or (lo is not None and default_value < cast(lo)) or (hi is not None and default_value > cast(hi)):
                raise ValueError(f"Default/step is outside valid bounds for camera parameter {key!r}")
            field["default"] = default_value
            field["step"] = step_value
        elif kind == "bool":
            bool_default = item.get("default", False)
            if not isinstance(bool_default, (bool, np.bool_)):
                raise ValueError(f"Boolean camera parameter {key!r} must have a boolean default.")
            field["default"] = bool(bool_default)
        else:
            field["default"] = str(item.get("default", ""))
        seen.add(key)
        validated.append(field)
    return validated


def _live_connection_options(profile: CameraProfile) -> tuple[str, ...]:
    """Return actual device connections, excluding the recorded-file workflow."""
    return tuple(option for option in profile.live_connections if option.casefold() != "recorded file")


def adapter_status(profile_id: str) -> dict[str, Any]:
    profile = get_profile(profile_id)
    plugins = discover_camera_adapters()
    entry = plugins.get(profile_id.casefold())
    adapter = None
    builtin_adapter = False
    load_error = None
    try:
        adapter = get_adapter(profile_id)
        if entry is not None and adapter is None:
            load_error = "Adapter entry point factory returned None."
        elif entry is None and adapter is not None:
            builtin_adapter = True
    except Exception as exc:  # a broken optional adapter must not prevent core file analysis
        load_error = f"{type(exc).__name__}: {exc}"
    methods = _adapter_method_names(adapter)
    has_live_connection = bool(_live_connection_options(profile))
    return {
        "profile": profile.label,
        "plugin_installed": entry is not None or builtin_adapter,
        "adapter_loaded": adapter is not None,
        "plugin_name": entry.name if entry else ("ThermoProp built-in adapter" if builtin_adapter else None),
        "adapter_origin": "third-party plugin" if entry is not None else ("built-in" if builtin_adapter else "not installed"),
        "plugin_load_error": load_error,
        "native_extensions": list(profile.native_extensions),
        "standard_extensions": list(profile.standard_extensions),
        "offline_sdk": profile.offline_sdk,
        "live_sdk": profile.live_sdk,
        "live_connections": list(profile.live_connections),
        "calibration_notes": profile.calibration_notes,
        "documentation_url": profile.documentation_url,
        "status_note": profile.status_note,
        "adapter_methods": sorted(methods),
        "native_decode_available": "load_recording" in methods,
        "live_probe_available": adapter is not None and "probe" in methods and has_live_connection,
        "single_camera_acquisition_available": adapter is not None and "acquire_sequence" in methods and has_live_connection,
        "live_acquisition_available": adapter is not None and bool({"acquire_sequence", "acquire_multi_sequence"} & methods) and has_live_connection,
        "multi_camera_acquisition_available": adapter is not None and "acquire_multi_sequence" in methods and has_live_connection,
        "camera_parameter_schema_available": adapter is not None and "get_parameter_schema" in methods,
        "configure_available": adapter is not None and "configure" in methods,
        "healthcheck_available": adapter is not None and "healthcheck" in methods,
    }


def probe_camera(profile_id: str, connection: str, **options) -> list[dict]:
    profile = get_profile(profile_id)
    if connection not in _live_connection_options(profile):
        raise ValueError(f"Connection {connection!r} is not a live device connection for {profile.label}.")
    adapter = get_adapter(profile_id)
    if adapter is None:
        raise RuntimeError(f"No thermoprop camera adapter is installed for {profile_id!r}. Install the vendor SDK and adapter plugin first.")
    probe = getattr(adapter, "probe", None)
    if not callable(probe):
        raise RuntimeError(f"The {profile_id!r} adapter does not implement probe().")
    result = probe(connection=connection, **options)
    devices = list(result or [])
    normalized = []
    for index, device in enumerate(devices):
        if not isinstance(device, dict):
            raise ValueError("Camera probe results must be dictionaries.")
        item = dict(device)
        item.setdefault("device_id", str(item.get("serial") or item.get("id") or index))
        item.setdefault("label", " · ".join(str(item.get(k)) for k in ("model", "serial") if item.get(k)) or f"Camera {index + 1}")
        normalized.append(item)
    return normalized


def healthcheck_camera(profile_id: str, device_id: str | None = None, **options) -> dict:
    """Run the installed adapter's non-destructive device / SDK readiness check."""
    adapter = get_adapter(profile_id)
    if adapter is None:
        raise RuntimeError(f"No adapter is installed for {profile_id!r}.")
    method = getattr(adapter, "healthcheck", None)
    if not callable(method):
        raise RuntimeError(f"The {profile_id!r} adapter does not implement healthcheck().")
    result = method(device_id=device_id, **options)
    if isinstance(result, bool):
        return {"ok": result}
    if not isinstance(result, dict):
        raise ValueError("Camera healthcheck must return a boolean or mapping.")
    return dict(result)


def configure_camera(profile_id: str, device_id: str | None = None, settings: dict | None = None, **options) -> dict:
    """Apply SDK-specific settings only when the installed adapter exposes configure()."""
    adapter = get_adapter(profile_id)
    if adapter is None:
        raise RuntimeError(f"No adapter is installed for {profile_id!r}.")
    method = getattr(adapter, "configure", None)
    if not callable(method):
        raise RuntimeError(f"The {profile_id!r} adapter does not implement configure().")
    result = method(device_id=device_id, settings=dict(settings or {}), **options)
    return dict(result or {})


def acquire_camera_sequence(
    profile_id: str,
    connection: str,
    *,
    device_id: str | None = None,
    duration_s: float = 30.0,
    frame_rate_hz: float = 3.0,
    settings: dict | None = None,
    **options,
) -> dict[str, Any]:
    """Capture a finite sequence through an explicitly installed vendor adapter.

    This intentionally uses a finite acquisition contract rather than pretending
    that an arbitrary Streamlit rerun is a reliable continuous camera stream.
    """
    if not np.isfinite(duration_s) or not 0.1 <= float(duration_s) <= 3600:
        raise ValueError("Capture duration must be between 0.1 and 3600 seconds.")
    if not np.isfinite(frame_rate_hz) or not 0.01 <= float(frame_rate_hz) <= 1000:
        raise ValueError("Requested frame rate must be between 0.01 and 1000 Hz.")
    profile = get_profile(profile_id)
    if connection not in _live_connection_options(profile):
        raise ValueError(f"Connection {connection!r} is not a live device connection for {profile.label}.")
    adapter = get_adapter(profile_id)
    if adapter is None:
        raise RuntimeError(f"No thermoprop camera adapter is installed for {profile_id!r}.")
    acquire = getattr(adapter, "acquire_sequence", None)
    if not callable(acquire):
        raise RuntimeError(f"The {profile_id!r} adapter does not implement acquire_sequence().")
    result = acquire(connection=connection, device_id=device_id, duration_s=float(duration_s),
                     frame_rate_hz=float(frame_rate_hz), settings=dict(settings or {}), **options)
    return _validated_recording(result, source=f"{profile_id} live acquisition")



def acquire_multiple_camera_sequences(
    profile_id: str,
    connection: str,
    device_ids: list[str],
    *,
    duration_s: float = 30.0,
    frame_rate_hz: float = 3.0,
    settings: dict | None = None,
    **options,
) -> dict[str, Any]:
    """Acquire synchronized finite sequences from an adapter's multi-camera API.

    The adapter must return ``{"recordings": {device_id: recording_dict},
    "metadata": {...}}``. Every recording is validated independently. This
    contract does not assume that independently-started cameras are synchronized;
    the adapter must provide true shared timestamps or document the limitation.
    """
    profile = get_profile(profile_id)
    if connection not in _live_connection_options(profile):
        raise ValueError(f"Connection {connection!r} is not a live device connection for {profile.label}.")
    ids = [str(v).strip() for v in device_ids if str(v).strip()]
    if len(ids) < 2 or len(ids) > 16 or len(set(ids)) != len(ids):
        raise ValueError("Multi-camera capture requires 2–16 unique device identifiers.")
    if not np.isfinite(duration_s) or not 0.1 <= float(duration_s) <= 3600:
        raise ValueError("Capture duration must be between 0.1 and 3600 seconds.")
    if not np.isfinite(frame_rate_hz) or not 0.01 <= float(frame_rate_hz) <= 1000:
        raise ValueError("Requested frame rate must be between 0.01 and 1000 Hz.")
    adapter = get_adapter(profile_id)
    method = getattr(adapter, "acquire_multi_sequence", None) if adapter is not None else None
    if not callable(method):
        raise RuntimeError(f"The {profile_id!r} adapter does not implement acquire_multi_sequence().")
    result = method(connection=connection, device_ids=ids, duration_s=float(duration_s),
                    frame_rate_hz=float(frame_rate_hz), settings=dict(settings or {}), **options)
    if not isinstance(result, dict) or not isinstance(result.get("recordings"), dict):
        raise ValueError("Multi-camera adapter must return a mapping under 'recordings'.")
    recordings = result["recordings"]
    missing = set(ids) - {str(k) for k in recordings}
    if missing:
        raise ValueError(f"Multi-camera adapter omitted requested devices: {sorted(missing)}")
    validated = {str(device_id): _validated_recording(recordings[device_id], source=f"{profile_id} multi-camera device {device_id}")
                 for device_id in ids}
    metadata_value = result.get("metadata", {}) or {}
    if not isinstance(metadata_value, dict):
        raise ValueError("Multi-camera metadata must be a dictionary.")
    synchronization_method = str(metadata_value.get("synchronization_method", "")).strip()
    if not synchronization_method:
        raise ValueError("Multi-camera adapter must document metadata['synchronization_method']; never assume independent cameras are synchronized.")
    missing_timestamps = [device_id for device_id, rec in validated.items() if rec.get("timestamps_s") is None]
    if missing_timestamps:
        raise ValueError(f"Every camera in a multi-camera capture needs timestamps; missing for {missing_timestamps}.")
    metadata_value = dict(metadata_value)
    metadata_value["synchronization_method"] = synchronization_method
    metadata_value["synchronization_verified_by_adapter"] = bool(
        metadata_value.get("synchronization_verified_by_adapter", False)
    )
    return {"recordings": validated, "metadata": metadata_value, "device_ids": ids}

def load_vendor_recording(profile_id: str, payload: bytes, filename: str, *,
                          settings: dict | None = None, **options) -> dict:
    adapter = get_adapter(profile_id)
    if adapter is None:
        profile = get_profile(profile_id)
        raise RuntimeError(
            f"{profile.label} native file {filename!r} needs a matching SDK adapter. "
            "Install the vendor SDK and a thermoprop.camera_adapters plugin, or export scalar radiometric frames to TIFF/NPY/NPZ/CSV. "
            "A palette image is not a valid substitute."
        )
    loader = getattr(adapter, "load_recording", None)
    if not callable(loader):
        raise RuntimeError(f"The {profile_id!r} adapter does not implement load_recording().")
    result = loader(payload=payload, filename=filename, settings=dict(settings or {}), **options)
    return _validated_recording(result, source=f"{profile_id} recorded-file decoder")


def profile_as_dict(profile_id: str) -> dict:
    return asdict(get_profile(profile_id))
