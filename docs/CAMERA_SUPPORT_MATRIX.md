# Camera support matrix

This document distinguishes **selectable metadata profiles**, **generic working transports**, and **vendor-specific drivers**. Selecting a brand in the Streamlit UI does not install its SDK or guarantee that the exact model can be controlled.

## What works without a vendor SDK

| Workflow | Working capability | Important boundary |
|---|---|---|
| Generic scalar file import | NPY, NPZ, TIFF, CSV, ZIP image sequences | Input must be numeric scalar arrays; RGB/pseudocolour is rejected. |
| Generic radiometric matrix JSON | Built-in JSON recording reader and configurable finite HTTP JSON matrix gateway | Endpoint/gateway must already expose numeric matrices. No vendor endpoint path is guessed. |
| GenICam / GenTL | Built-in Harvester discovery, finite scalar-component capture, and opt-in ExposureTime/Gain/AcquisitionFrameRate writes when standard nodes are available | Requires the exact manufacturer's CTI producer; output is camera counts unless separately decoded/calibrated. Camera node ranges are checked and values are read back. |
| Third-party vendor adapter | Entry-point plugin contract | Vendor SDK, license and exact model support are supplied separately. |

## Selectable vendor and device profiles

The following profiles are available in the application. The notes describe the intended integration route, not a claim that every model has a supported live driver.

| Profile | Families / example products | Typical route |
|---|---|---|
| Generic / research dataset | Public datasets, lab exports, custom scalar matrices | NPY / NPZ / TIFF / CSV / ZIP |
| FLIR / Teledyne FLIR | Science, A-series, Boson, Tau, Lepton, supported machine vision | Science File SDK for supported files; Science Camera SDK or Spinnaker for supported live cameras; selected cores may use flirpy. Exact SDK depends on model. |
| InfraTec | ImageIR, VarioCAM, IRB | IRBACS for stored IRB data; IRBGRAB for online acquisition; license/add-on and model support apply. |
| Optris | PI / Xi series | OTC SDK for supported camera families; check Python/NumPy ABI and hardware connection. |
| Seek Thermal | Compatible OEM Core / Mosaic / Micro Core | Seek SDK 4.x and supported OEM core; not every consumer attachment is exposed. |
| HIKMICRO | Handheld and industrial/network models | Vendor-supported radiometric export or documented numeric matrix API; model/firmware semantics matter. |
| Hikvision | Thermographic and radiometric IP models | Model-specific SDK/ISAPI numeric endpoint or vendor export; palette RTSP is not radiometric. |
| Testo | Multiple handheld / fixed product families | Model-supported radiometric export/API; no universal native decoder assumed. |
| Fluke | Multiple handheld / fixed product families | Vendor-approved radiometric export or model-specific SDK/adapter. |
| InfiRay / Infisense | OEM cores, handheld and fixed systems | OEM SDK/export varies by core and distribution agreement. |
| Guide Sensmart | Handheld and fixed systems | Model-specific SDK / scalar export. |
| Workswell | WIRIS and thermoIMAGER integrations | Model-specific API / export; standard network video may be non-radiometric. |
| Xenics / Exosens | SWIR, MWIR, LWIR scientific detectors | Xeneth / model-specific SDK or compatible GenTL. Radiance/counts may need calibration. |
| Telops | Scientific high-speed / spectral systems | Vendor acquisition software/SDK, Camera Link frame grabber or model-specific export. |
| Sierra-Olympic | Infrared OEM modules and integrated cameras | OEM module SDK / documented data output. |
| Axis | Network thermal / radiometric camera families | Documented numeric API or custom network adapter. Ordinary thermal video may be a palette stream. |
| MOBOTIX | Thermal and dual-thermal systems | Documented model/firmware API or custom adapter. Check whether numeric radiometric arrays are actually exposed. |
| Basler | Infrared / machine-vision models | pylon or compatible CTI GenTL producer; camera counts do not equal °C. |
| Allied Vision | Infrared / machine-vision models | Vimba X or compatible CTI GenTL producer; sensor-specific calibration remains separate. |
| IDS | IDS peak / uEye compatible systems | Vendor SDK or GenTL producer; check exact camera family and transport. |
| LUCID Vision Labs | Triton / Atlas and other compatible systems | Arena SDK or compatible GenTL producer; radiometry is model-specific. |
| Teledyne DALSA | Scientific / machine-vision infrared systems | Sapera LT, GenTL, or Camera Link frame-grabber adapter. |
| XIMEA | Infrared / SWIR / hyperspectral cameras | xiAPI or compatible model-specific interface; spectral cubes require a dedicated import route. |
| Raptor Photonics | SWIR / MWIR / LWIR scientific detectors | Vendor SDK / Camera Link / frame grabber; often requires detector-specific calibration. |
| New Imaging Technologies (NIT) | SWIR and other infrared camera families | Model-specific SDK / numeric export; check output units. |
| Smartphone thermal attachments | Compatible FLIR / Seek and other OEM accessories | Mobile/OEM SDK and OS/model-specific support; screenshots are not radiometric data. |
| Thermopile arrays / low-resolution thermal sensors | MLX90640-class and other sensor arrays | I²C / SPI / serial / DAQ adapter or scalar matrix export; calibration and ambient compensation are sensor-specific. |
| GenICam / GenTL industrial cameras | Basler, Allied Vision, IDS, LUCID, Teledyne DALSA, other CTI-compatible manufacturers | Built-in Harvester transport when a compatible CTI Producer is installed. |
| Generic HTTP / ISAPI matrix API | Network cameras or a local lab gateway | Configured numeric JSON endpoint. No undocumented ISAPI path or RTSP palette decode. |
| Other / custom | Any unlisted vendor, OEM module or future camera | Scalar import, supported GenTL/HTTP path, or a third-party ThermoProp adapter. |

## Detector architecture fields in the UI

The optional “Camera / detector architecture” expander records:

- Physical interface: USB, USB-C, USB3 Vision, GigE Vision, 10 GigE, Wi-Fi, HTTP/REST/ISAPI, RTSP/RTP (visual-only unless numeric pixels are documented), GenICam/GenTL, Camera Link, Camera Link HS, CoaXPress, PCIe/frame grabber, serial/UART, I²C/SPI and custom OEM interfaces.
- Form factor: handheld, fixed, high-speed scientific, machine-vision, network/IP, OEM core, smartphone, UAV, thermopile array, multispectral/hyperspectral, line-scan, microscopy and other.
- Detector: uncooled microbolometer, cooled photon detector, thermopile, InSb, MCT/HgCdTe, InGaAs, QWIP, pyroelectric, CMOS/CCD, hybrid and other.
- Spectral band: NIR, SWIR, MWIR, LWIR, VLWIR, multiband/hyperspectral or unknown.
- Output representation: calibrated temperature, radiance, linearized counts, raw counts, relative intensity, or false-colour display image.
- Calibration state and pixel format.

These fields are provenance only. They do not calibrate a matrix, decode an SDK payload, or activate a connection. “Unknown / not documented” is the correct choice when the camera datasheet or export metadata does not state the value.

## Calibration and transport cautions

1. A 16-bit image is not automatically a temperature map. It may be raw counts, linearized counts, radiance, or a vendor-specific fixed-point mapping.
2. A pseudocolour RGB stream cannot be converted back to accurate temperatures from its palette alone. Use the original radiometric export or a vendor endpoint exposing numeric matrix values.
3. GenTL standardizes device transport/acquisition interfaces, not all camera-specific radiometric correction algorithms.
4. Multi-camera synchronization requires a shared hardware trigger or documented synchronized timebase. Independent host capture times do not prove synchronization.
5. Proprietary vendor SDKs, CTI producers, licenses, drivers, and hardware are not bundled in this project.
6. This software is research-only and not a medical device. Thermal-front propagation is not vascular direction or blood-flow velocity.

## Official and primary integration resources

- [FLIR SDK selection guide](https://flir.custhelp.com/app/answers/detail/a_id/3000/~/flir-cameras---which-sdk-should-i-use-for-my-camera-application%253F)
- [FLIR Science Camera SDK](https://flir.custhelp.com/app/answers/detail/a_id/6623/~/science-camera-sdk---download)
- [FLIR Science File SDK](https://flir.custhelp.com/app/answers/detail/a_id/6622/related/1)
- [InfraTec SDKs](https://www.infratec.eu/thermography/thermographic-software/sdk/)
- [Optris OTC SDK](https://optris.github.io/otcsdk_downloads/)
- [Seek Thermal Python bindings](https://github.com/seekthermal/seekcamera-python)
- [EMVA GenICam](https://www.emva.org/standards-technology/genicam/)
- [Harvester GenTL Python library](https://harvesters.readthedocs.io/)
- [HIKMICRO industrial cameras](https://www.hikmicrotech.com/en/industrial-products/furnace-cameras/)

Consult the exact camera model, firmware, SDK version, OS support and licensing page before installing a driver.
