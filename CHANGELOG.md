# Changelog

## v1.5.0 — broad camera catalog and generic transport acquisition

- Expanded the selectable camera catalogue across radiometric handheld, fixed/network, scientific, machine-vision, OEM/mobile, thermopile and multispectral camera families, including an Other/custom profile.
- Added user-selectable camera form factor, detector technology, spectral band, output representation, calibration state and pixel-format metadata. These are provenance fields, not drivers or calibration operations.
- Clarified that absence of a vendor adapter does not block generic scalar file import; only native decoding and live operations require an adapter.
- Added built-in GenTL generic capture through Harvester plus a configurable HTTP JSON radiometric-matrix gateway. GenTL outputs remain camera counts unless a separate calibrated decoder is used; HTTP endpoints are operator-configured and no vendor ISAPI path is guessed.
- Corrected GenTL timestamp handling so device timestamps and host receive times are never mixed; record the selected time source and reject non-monotonic clocks.
- Added opt-in generic GenICam ExposureTime, Gain and AcquisitionFrameRate settings with node availability/writability/range checks and read-back metadata.
- Improved export secret redaction, including sensitive query parameters embedded in configured endpoint URLs.
- Added camera support matrix and a sample JSON matrix recording.

## v1.4.0 — configurable device and sensor integrations

- Added capability-aware camera vendor profiles and a Python entry-point adapter contract.
- Added per-camera dynamic SDK parameter schemas, optional probe/health/configuration hooks, finite capture validation, and native recording loader hooks.
- Added single- and multi-camera capture boundaries with required timestamps and explicit synchronization metadata. Heterogeneous vendor capture requires a custom orchestration adapter; camera records remain separate and are not silently fused.
- Added LSL stream discovery/capture with UID selection, bounded finite capture, timestamp and clock-correction metadata, and sample-quality diagnostics.
- Added XDF stream/channel import with independent clock synchronization and de-jitter controls; de-jitter is off by default.
- Added optional SHAP global explanations for supported tree-based reference models.
- Retained participant-grouped reference regression, optional thermal-domain YOLO ROI suggestions, wavelet analysis, MLflow/DVC hooks and Pennes forward simulation.
- Added capability and validation safeguards so selecting a vendor/sensor profile is not presented as a hardware connection or a calibrated measurement.

## v1.2.0 — interactive analysis workspace

- Added browser-side animated thermal frame viewer, protocol-aware timeline, polygon/rectangle ROI editor, named ROI analysis, multi-ROI metrics, and improved QC/export tools.
- Added separate baseline-relative and event-relative thermal progression channels for recordings with unrecorded challenge intervals.

## v1.4.1 — optional ML dependency validation

- Defer scikit-learn imports until grouped-reference training inputs pass schema, participant-count and sample-count validation.
- Return an actionable installation message for valid ML training requests when the optional `ml` extra is missing.
- Add regression coverage for the missing optional dependency path.
- Document `pip install -e ".[ml]"` and clarify that the grouped-ML training test is skipped in a core-only installation.
