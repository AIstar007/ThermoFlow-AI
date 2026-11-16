# Technical report — Thermal Pattern Propagation Lab v1.5.0

## 1. Objective and scientific scope

The project processes scalar thermal-image sequences to quantify surface-temperature variation over time, identify warming/cooling regions, estimate apparent thermal-front propagation, and summarize the dynamics inside named regions of interest (ROIs).

The output is not a direct measurement of blood flow. Infrared thermography observes emitted radiation and, after suitable radiometric calibration, estimates skin-surface temperature. Surface temperature is influenced by perfusion, conduction, convection, radiation, evaporation, room conditions, emissivity, camera calibration, motion, and anatomy. A thermal-front direction or speed is therefore an image-derived descriptor, not blood-flow direction or velocity.

## 2. System architecture

- **Python analysis layer:** NumPy, SciPy, pandas, scikit-image and tifffile.
- **Interactive application:** Streamlit 1.67+.
- **Browser-side ROI drawing:** Streamlit Components v2 with a canvas editor for polygon and rectangle geometry.
- **Interactive figures:** Plotly 6, including a browser-side animated heatmap, phase-annotated ROI time series, threshold-sensitivity table, and registration-motion plot.
- **Reproducibility:** CSV, JSON, NPZ, PNG, standalone Plotly HTML, and ZIP export.

Streamlit is retained because image ingestion, numerical processing and experiment metadata are already Python-based. Components v2 supplies custom browser interactions without requiring a separate React service or duplicating application state. Plotly handles the scientific plots and animation. A migration to Next.js/React is not currently justified; it would become more attractive for continuous live-camera acquisition, multi-user concurrent sessions, or a highly customized workstation interface.

## 3. Input data and unit handling

The application accepts scalar sequences in `(frames, height, width)` form from:

- NPY arrays;
- NPZ archives (the array key is selectable when multiple arrays are present);
- multi-page TIFF;
- ZIP archives containing one ordered grayscale image sequence.

RGB/pseudocolour images are rejected because their display colours are not temperature values. The application requires the operator to label values as °C, relative intensity, or camera counts. Selecting °C is only a label; a separate confirmation records the operator's statement that radiometric calibration is present. The application does not convert ordinary PNG intensity values to Celsius. Vendor-native radiometric file formats may need a vendor-specific export/parser before ingestion.

A timestamp CSV can be supplied. It must contain one finite, strictly increasing time value per frame. Otherwise, a nominal frame rate is used. Frame-interval coefficient of variation and large gaps are reported as engineering QC indicators.

## 4. Protocol-aware timebase

Two recording protocols are supported.

### 4.1 Baseline + rewarming, challenge interval not recorded

This is the relevant mode for DIRT-SR38. The released sequence has a pre-cooling steady-state baseline followed by a separate rewarming recording. The cooling stimulus interval is between the recorded sections and is not itself present as a frame phase.

The application labels the initial frames `Baseline` and the subsequent frames `Rewarming · challenge not recorded`. It inserts a derivative discontinuity at the boundary. If the missing challenge duration is independently documented, the operator may include that duration in the displayed timebase. A timestamp discontinuity is not automatically assumed to be the cooling duration; the operator must confirm that interpretation.

### 4.2 Continuous baseline → challenge → recovery

This mode is for a recording in which all three phases are actually present. The operator supplies challenge start and exclusive end frame. The UI does not create a synthetic challenge segment for a recording that lacks it.

For both modes, supplied timestamps are preserved. The time-axis label indicates whether a gap is omitted, documented, or simply an unverified timestamp discontinuity.

## 5. Preprocessing and motion handling

The sequence is converted to float32 while preserving numeric scale. Invalid pixel values are handled by the preprocessing layer. Spatial Gaussian denoising is configurable. Optional translational phase-correlation registration aligns frames to a baseline reference; it is off by default because image registration can suppress genuine moving thermal patterns or introduce distortions. The UI plots estimated x/y shifts so users can inspect the correction rather than accepting it invisibly.

Temporal smoothing is applied independently inside continuous recorded segments. It does not blur across a known unrecorded challenge interval. Temporal derivatives are also computed independently inside each continuous segment and remain undefined at a one-frame segment boundary where there is insufficient support.

## 6. ROI editor and regional measurements

The browser-side editor supports named polygons and rectangles, colour labels, undo, clear, and full-frame selection. Vertices are stored in source-image pixel coordinates. Python validates polygon size, image bounds, unique IDs/names and rasterization. Overlapping ROIs are allowed but reported; their areas must not be summed as if they were disjoint.

If no anatomical ROI is drawn, the app analyses one clearly labelled `Full frame` region without writing that default back into the editable ROI list. This keeps analysis and exports usable while making the no-ROI choice visible.

All named ROI measurements are extracted from a single full-resolution analysis pass. For each frame and ROI, the app records mean, median, standard deviation, baseline-relative change, event-relative change, mean temporal derivative, warming/cooling fractions, and ROI pixel count.

## 7. Baseline-relative and event-relative change

The per-pixel pre-protocol baseline is:

\[
T_0(x,y)=\operatorname{median}_{t\in B} T(x,y,t),
\]

where \(B\) denotes the operator-selected baseline frames. The baseline-relative change is:

\[
\Delta T_0(x,y,t)=T(x,y,t)-T_0(x,y).
\]

This channel shows the difference from the pre-protocol state and is retained for reference.

A separate **event-relative** reference is essential for rewarming recordings. After an unrecorded cooling challenge, the skin may warm during the recorded recovery phase while remaining below its pre-cooling baseline. A detector that only searches for positive values of \(\Delta T_0\) can miss this warming response.

The app therefore takes the median of a short reference window beginning at the first recorded rewarming frame (or the recorded challenge start in continuous mode). The default is one frame; the UI allows a 1–10 frame median window to trade noise reduction against blurring a fast front:

\[
T_e(x,y)=\operatorname{median}_{t\in E}T(x,y,t),\qquad
\Delta T_e(x,y,t)=T(x,y,t)-T_e(x,y),
\]

where \(E\) is the selected event-reference window. Frames before the selected event reference are masked from the event-progression channel. The UI shows baseline-relative and event-relative maps separately. The selected event reference frame/window is recorded in the summary and export.

## 8. Threshold and propagation estimation

Robust baseline noise is estimated from median absolute deviation:

\[
\sigma_{robust}=1.4826\,\operatorname{median}(|r-\operatorname{median}(r)|).
\]

The applied threshold is the maximum of the requested threshold, a minimum noise floor, and three times the robust baseline noise estimate. It is in the input units and requires dataset-specific review.

For each pixel, the first time at which event-relative change crosses a positive or negative threshold is recorded. The spatial gradient of a smoothed arrival-time map gives a local normal direction toward later threshold arrival. Its inverse gradient magnitude provides an apparent normal speed:

\[
\mathbf n=\frac{\nabla t_a}{\|\nabla t_a\|},\qquad
v_n=\frac{1}{\|\nabla t_a\|}.
\]

With no documented pixel pitch, speed is reported in px/s. With an isotropic pixel pitch, it is reported in mm/s. The optional scale must come from camera geometry/calibration; it is not inferred from the PNG dimensions. The direction is the direction of the thermal threshold front, not the direction of blood flow.

The UI also varies the threshold multiplier and reports how much of the selected ROI is crossed. This is a sensitivity analysis, not an accuracy metric.

## 9. Recovery features and quality control

An optional exponential fit is calculated separately for each ROI over the configured recovery segment:

\[
T(t)=T_{\infty}-A\exp(-(t-t_0)/\tau).
\]

The fitted \(\tau\) and \(R^2\) are descriptive curve parameters only. They are not perfusion units or clinical outputs.

Engineering QC indicators include finite-sample fraction, ROI coverage, timestamp variability, baseline median drift normalized to the sequence interquartile range, a bounded-memory sampled fraction beyond global 3×IQR fences, registration shift magnitude, and ROI overlap. A warning is raised for large frame-interval variation, low ROI coverage, drift above the configured scale-relative rule, sampled outliers above the configured fraction, large translations, or overlapping named regions. These indicators help identify input or acquisition concerns; they are not clinical quality grades or a validated QC score.

## 10. Real dataset plan: DIRT-SR38

The recommended public sequence dataset is DIRT-SR38 (CVPR Workshops 2026 / Zenodo DOI [10.5281/zenodo.19241293](https://doi.org/10.5281/zenodo.19241293)). The paper describes 38 healthy adult female participants. Each subject has a 30-second steady-state baseline (90 frames) and a four-minute rewarming sequence (720 frames), totaling 810 aligned LR/HR frame pairs. The cooling stimulus is between the recorded baseline and recovery sequences; it is not supplied as a challenge-phase video segment.

The released PNGs are intensity-normalized relative values, not calibrated Celsius temperatures. The LR images are 108×108 and HR images 216×216. The paired LR release is sampled at 3 Hz. The nominal 150×150 mm ROI corresponds approximately to 1.389 mm/px for 108×108 LR and 0.694 mm/px for 216×216 HR, subject to checking crop geometry and any preprocessing metadata before use.

Recommended initial settings:

- use one participant and one modality per analysis;
- concatenate only that participant's 90 baseline frames with the matching 720 rewarming frames, in temporal order;
- select `Baseline + rewarming (challenge interval not recorded)`;
- label values `relative intensity`, never °C;
- use 3 Hz only after confirming the released sequence frame cadence;
- set 90 baseline frames;
- set the corresponding documented pixel pitch only if the image crop has the stated geometry;
- record the unobserved stimulus duration as unknown unless an independent protocol record supplies it;
- compare sensitivity to event-relative threshold and smoothing settings.

The archive is approximately 3.6 GB and is not bundled. No real DIRT-SR38 participant recording was processed for this release. The two bundled example sequences are synthetic software tests.

## 11A. v1.5 camera catalogue and generic connection paths

Version 1.5 expands selectable profiles across major thermography vendors, network/IP cameras, scientific SWIR/MWIR/LWIR systems, machine-vision manufacturers, smartphone OEM cores, thermopile arrays, GenICam/GenTL and custom instruments. The UI separately captures physical interface, form factor, detector technology, spectral band, output representation, calibration status and pixel format. These fields are instrument metadata only; they do not enable a hardware driver or calibrate a detector output.

Two generic connection paths are built in: (1) Harvester-based GenTL discovery and finite scalar-component capture when the optional vision extra and exact manufacturer's CTI producer are installed; (2) a configurable HTTP JSON radiometric matrix reader/gateway for a documented numeric matrix endpoint. Generic GenTL values remain counts unless a model-specific decoder is applied. The HTTP connector does not guess ISAPI/REST endpoints or decode ordinary RTSP palette video. Native vendor decoding and direct hardware control still require separately installed, licensed, model-specific adapters. See `docs/CAMERA_SUPPORT_MATRIX.md` for the selectable profiles and exact boundaries.

## 11. Synthetic software validation

Two deterministic 64×64 sequences with 80 frames at 2 Hz contain known planar warming/cooling fronts with expected speed 4 px/s. The initial validation outputs are approximately 4.027 px/s for rightward warming and 4.031 px/s for leftward cooling. Tests also cover event-relative warming below the pre-challenge baseline, no smoothing across a known protocol gap, timestamp validation, polygon geometry, overlap, ROI metrics, QC, and Plotly chart construction.

The v1.5.0 test suite contains 84 tests. Synthetic agreement validates the implementation against programmed examples only. It does not establish physiological accuracy, generalization, or perfusion validity.

Run:

```bash
python examples/run_synthetic_demo.py
pytest -q
streamlit run app.py
```

## 12. Remaining limitations and validation needed

- No real participant sequence has been processed in this build.
- DIRT-SR38 has no synchronized laser-speckle/laser-Doppler perfusion ground truth in the released thermal PNG sequence and no absolute °C calibration.
- The cooling interval is absent from DIRT-SR38's released frames; recovery time zero is the first recorded rewarming frame unless independent timestamps document otherwise.
- A simple planar-front test is much easier than the spatially heterogeneous, diffuse and non-planar patterns of real skin thermoregulation.
- Translation-only registration does not correct rotation, perspective, deformation, anatomical movement or non-rigid motion.
- ROI contours are manually drawn; automatic segmentation/anatomical tracking is not included.
- The app provides finite adapter-based live thermal capture when a matching vendor SDK adapter is separately installed. No proprietary vendor adapter is bundled or hardware-certified in this release; camera-specific operation still needs a tested adapter for the exact model, firmware, OS and SDK.
- The Equation 10 heat-balance implementation from the 2023 reference remains an optional CLI/pipeline model path; it is not exposed as a default dashboard measurement and requires calibrated Celsius input and ambient temperature. It must not be used with DIRT-SR38 normalized PNGs.
- This is research software, not a medical device, and no diagnosis or clinical perfusion decision is supported.

A defensible physiological validation study should acquire radiometric thermal sequences and synchronized independent perfusion data (e.g., LSCI or laser Doppler flowmetry) using a standardized protocol. Compare ROI-level recovery features, lag/correlation, test-retest reliability and agreement. Split and analyse at the participant level, not by randomly distributing correlated frames across train/test. If the intended use is vascular screening, also compare with prespecified clinical reference measures and an independent cohort/camera.

## 13. Assignment deliverable status

| Assignment item | Status |
|---|---|
| Research papers, data and repositories | Reviewed and documented |
| Thermal input preprocessing | Implemented for scalar arrays and common exports |
| ROI definition | Interactive named polygon and rectangle editor; full-frame fallback |
| Frame-by-frame temporal analysis | Implemented |
| Warming/cooling regions | Event-relative threshold crossing maps and ROI features |
| Apparent direction and propagation | Implemented; outputs explicitly not blood-flow velocity |
| Visualization | Animated Plotly heatmap, arrival maps, vectors, ROI time series and QC |
| Complete source code | Included |
| Two different sequences | Two synthetic software-validation sequences; real subject data still to be processed |
| Technical report | Included |
| Working demonstration | CLI synthetic demos and app code included; browser launch is not runtime-verified in this offline build environment |
| Physiological/perfusion validation | Pending synchronized independent perfusion reference data |

## 14. Conclusion

The v1.2/v1.4 development line improves both the interface and the scientific workflow. The most consequential algorithmic change is the separate event-relative reference for recovery sequences, so rewarming is not missed merely because skin temperature remains below its pre-challenge baseline. Protocol gaps are kept explicit, smoothing and temporal derivatives do not bridge known unrecorded intervals, and named regional measurements are computed from one full-resolution analysis pass.

The next milestone is a real-data run on at least two DIRT-SR38 participants, followed by independent perfusion-reference validation. Adding a neural network before obtaining defensible labels and reference data would not improve the scientific validity of the result.


## 11. Optional integration architecture (v1.3 foundation, extended in v1.4)

The v1.3 foundation and v1.4 extension provide an extensible integration layer. Camera profiles are capability descriptions, not universal camera drivers. Standard scalar arrays and radiometric exports can be imported using the portable readers. Proprietary file decoding and live device control require a supported vendor SDK and an installed `thermoprop.camera_adapters` plugin. The repository includes the adapter contract and template; it does not redistribute proprietary SDKs or claim that every vendor model has a tested driver.

Reference-sensor CSV channels can be aligned using user-selected timestamp/value columns and an explicit clock offset. The alignment path does not extrapolate beyond the recorded signal range. Lag analysis interpolates on actual timestamps and splits the search at known protocol derivative boundaries. Pearson/Spearman values, p-values and lag are descriptive; serial dependence and multiple comparisons are not automatically corrected. LSCI and laser-Doppler are treated as preferred independent local perfusion reference channels when collected appropriately. PPG, ECG and EDA are complementary physiology, not local perfusion ground truth.

Optional thermal-domain YOLO segmentation weights can propose ROI polygons. The model is user-supplied; generic RGB weights are not presumed to generalize to thermal images. All proposals require manual review.

The grouped reference-model studio accepts participant/session/ROI summary rows and uses participant-grouped out-of-fold validation. It includes ElasticNet, tree ensembles, HistGradientBoosting and optional XGBoost. It requires at least six distinct participant groups and at least 12 rows; this is a software guard, not evidence that the sample is sufficient for a validated clinical model. External cohort/site validation is still mandatory. Model predictions are limited to the independent reference target supplied by the user.

The Pennes module is a configurable one-dimensional homogeneous slab forward simulator with an explicit time step and idealized boundary conditions. It is a sensitivity and teaching tool only. No inverse solver is used to infer a real participant's perfusion from surface thermography. The optional Eq. (10) map implements the supplied 2023 paper-derived heat-balance expression and is gated behind explicit confirmation of radiometrically calibrated Celsius input. Its assumptions and parameter dependence remain important; the output is a model-derived index, not direct flow.

## 12. Reproducibility and software validation

The export ZIP includes `manifest.json` with source and timestamp hashes, selected camera profile, camera identifiers and calibration ID, unit label, calibration confirmation, protocol mode, registration settings and software versions. It does not embed the raw source recording or original sensor CSVs. Preserve those source files in controlled storage with the manifest. Optional MLflow integration logs settings, summary and selected numerical metrics. A DVC stage (`dvc.yaml`) reproduces the synthetic validation outputs; DVC remains an optional data-versioning dependency and must be configured with a private, approved remote for research data.

The 84-test v1.5.0 suite covers sensor parsing/alignment, timestamp-aware lag estimation across protocol breaks, expanded vendor profile catalogues, detector/physical-interface metadata options, camera adapter loading and dynamic parameter validation, generic HTTP matrix decoding, GenTL probing/capture, timestamp-source fallback and optional GenICam exposure/gain/frame-rate node validation, multi-camera timestamp/synchronization requirements, XDF and LSL integrity, SHAP explanations when available, affine registration, grouped ML validation guards, Welch/CWT, Pennes constraints, and imaging/ROI/protocol/QC/export tests. Tests on synthetic inputs validate software behaviour only; they do not validate the physiological interpretation.


## 12. v1.4 optional device and sensor integration architecture (historical)

Version 1.4 adds a capability-aware camera adapter interface, adapter-defined parameter controls, health checks, camera configuration and finite single-/multi-camera capture. The adapter must validate and return scalar frames and timestamps. Multi-camera adapters must declare a synchronization method and return timestamps for each device; synchronization is not marked verified unless the adapter explicitly says so. The app retains per-camera recordings and analyses a selected device sequence. It does not assume independent cameras share a clock; mixed vendor capture requires a custom orchestration adapter.

The optional sensor layer accepts CSV, LabRecorder XDF and finite live LSL captures. XDF clock synchronization is enabled by default; dejittering is a separate user option and defaults off to preserve irregular sampling and dropped-frame gaps. LSL clock correction maps the outlet to the receiving computer's LSL clock and does not correct a thermal camera clock unless that clock is itself synchronized or a measured offset is entered.

The supervised model studio includes optional SHAP global explanations for supported tree regressors. Explanations describe model feature attribution on the provided cohort; they are not causal effects. Participant-grouped validation and independent perfusion-reference labels remain necessary.

Vendor SDKs remain external dependencies. Camera profile selectors do not install drivers or guarantee a connection. Exact support depends on camera model, firmware, SDK version, operating system, license and the separately installed `thermoprop.camera_adapters` plugin. See `docs/INTEGRATIONS.md` and the downloadable adapter template for the current contract.


## 13. v1.4 device configuration and sensor connectivity (historical)

The sidebar offers generic, FLIR, InfraTec, Optris, Seek, HIKMICRO, Testo, Fluke, InfiRay/Guide, GenICam/GenTL, and custom camera profiles. The chosen profile controls which connection workflows and recorded native extensions are offered. The selected adapter may publish a parameter schema (booleans, numbers, text and enumerated selections); the UI renders those controls and passes values to that adapter. The profile itself is not a driver: the vendor SDK, exact camera support, plugin and any license are external. When an adapter is absent or fails to load, file-based scalar import remains available.

The adapter can optionally enumerate devices, run a non-destructive health check, configure the camera, capture one finite sequence, capture multiple device sequences, and decode supported native files. Multiple camera recordings remain separate; the user selects one sequence to analyse. A multi-camera adapter must provide timestamps for every device and declare a synchronization method. Synchronization is marked verified only if the adapter explicitly confirms that status.

The reference sensor layer accepts CSV, XDF and finite live LSL channel capture. XDF clock synchronization and timestamp de-jitter are independent controls; de-jitter is off by default. LSL streams are resolved by unique stream UID when available, with source ID fallback. Inlet clock correction maps the outlet clock into the receiving LSL clock; an independent thermal-camera clock still requires a measured offset or common trigger. Reordered samples, duplicate timestamps, invalid samples and timing gaps are reported rather than silently converted into a continuous physiological trace.

Optional SHAP explanations are available for the supported tree regressors. Grouped participant validation is maintained. No perfusion estimator is enabled without paired independent reference labels.
