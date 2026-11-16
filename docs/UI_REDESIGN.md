# UI architecture and interaction redesign — v1.2.0

## Decision: Streamlit + Components v2 + Plotly

The project remains a Python research application. A full migration to React/Next.js or a new framework is not currently justified: the analysis pipeline, input handling, protocol configuration, and export workflow are already Python-native. The difficult interactions can now be added through Streamlit Components v2 while retaining one deployable app.

The current Streamlit documentation recommends Components v2 for new custom components. Components v2 run integrated into the page (without the legacy iframe isolation) and support state and trigger callbacks in both directions between browser JavaScript and Python. This is a good fit for the named polygon/rectangle ROI annotator. See:

- [Custom components v2](https://docs.streamlit.io/develop/concepts/custom-components/components-v2)
- [Component registration](https://docs.streamlit.io/develop/concepts/custom-components/components-v2/register)
- [Component mounting](https://docs.streamlit.io/develop/concepts/custom-components/components-v2/mount)
- [Bidirectional communication](https://docs.streamlit.io/develop/concepts/custom-components/components-v2/communicate)

The application pins `streamlit>=1.67,<2` to ensure this API is available.

## What is implemented

### 1. Animated sequence explorer

- Plotly heatmap animation with Play/Pause, frame slider and playback speed.
- Raw, baseline-relative Δ, event-relative progression, and dT/dt channels with phase labels in frame titles.
- Fixed robust color limits across frames to prevent automatic contrast changes from looking like physical dynamics.
- Named ROI contours stay visible during the animation.
- Browser preview is decimated for responsiveness; analysis remains at original array resolution.

Plotly documents frame animation, slider controls and play/pause buttons in its [Python animation guide](https://plotly.com/python/animations/). Streamlit exposes Plotly figures through [`st.plotly_chart`](https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart).

### 2. Direct ROI annotation

The custom Components v2 canvas editor supports:

- named polygon ROIs with arbitrary vertices;
- rectangle ROIs built from two opposite corners;
- per-region names and colours assigned at drawing time;
- undo, finish shape, full-frame preset, and clear-all actions;
- coordinates retained in source-image pixel space;
- validation for duplicate names/IDs, too-small polygons, non-finite points and out-of-frame coordinates;
- overlap reporting and separate per-ROI metrics.

The component is registered once per Python process and exchanges the named region list through `setStateValue("regions", ...)`.

### 3. Protocol-aware timeline

Two modes are explicitly represented:

- Continuous baseline → recorded challenge → recovery.
- Baseline + rewarming when the challenge/cooling interval is not recorded.

The second mode introduces a derivative discontinuity at the baseline boundary. A finite difference is never calculated across an unobserved cooling interval. When gap duration is unknown, the time axis is labelled as recorded time with the gap omitted. A gap can only be included when the user explicitly confirms independently documented timing or supplies timestamps that span that interval.

The DIRT-SR38 paper describes 90 baseline frames and 720 rewarming frames per participant, with the cooling stimulus between those recorded phases. The stimulus interval is not an image phase. The app therefore does not invent a cooling segment in DIRT-SR38.

### 4. Multiple ROI comparison

The sequence is preprocessed and analysed once over the union of selected ROIs. Each named ROI then receives separate metrics extracted from the shared baseline, Δ and derivative arrays. This avoids recalculating the full 3-D sequence transform for every ROI.

### 5. QC, motion and robustness

- timestamp/frame interval coefficient of variation and maximum interval;
- finite-pixel fraction;
- baseline median drift in input units/minute and relative to the sequence IQR;
- a bounded-memory, spatial-temporal sample of the fraction beyond global 3×IQR fences;
- ROI coverage and overlap;
- translational registration shift trace when registration is enabled, with a warning for unusually large shifts;
- warming/cooling coverage at alternative threshold multipliers.

These are engineering indicators, not a clinical quality score.

### 6. Export and reproducibility

The ZIP bundle includes summary JSON, union and regional metric CSVs, protocol timeline CSV, ROI geometry, compressed arrays, static figure, and a standalone interactive Plotly player HTML. Raw source data and original radiometric calibration files are deliberately not embedded; retain those separately.

## Why not switch to Gradio, Panel, or React now?

| Framework | Strength for this use case | Trade-off | Decision |
|---|---|---|---|
| Streamlit + Components v2 + Plotly | Python-native workflow, session state and uploads, bidirectional browser component, scientific plots and browser playback | Components have to be maintained and tested against Streamlit API versions | **Selected** |
| Gradio Blocks | Event-driven Python interface and ready-made image/video components | A different interaction/state model; does not remove need for a bespoke anatomical ROI tool | Not selected |
| Panel / HoloViz | Rich reactive widgets and scientific visualization ecosystem | Broader migration and deployment footprint | Future alternative if this becomes a long-running scientific visualization workstation |
| React/Next.js + FastAPI | Maximum pointer interaction, WebSocket streaming, multi-user annotation, fine-grained frontend state, authentication | Separate frontend/backend, API, deployment and maintenance | Future product architecture if live camera streaming, many concurrent users or complex collaboration are required |

## Remaining high-value work

1. **Real-data validation:** process at least two DIRT-SR38 participants, LR and HR separately, inspect timestamps and compare regional recovery curves. This is the immediate next step.
2. **Radiometric vendor adapters:** implement one camera-specific parser once the target camera/file format is known. Preserve original calibration metadata.
3. **Registration quality:** add overlay review and a confidence metric; translation-only phase correlation is not appropriate for deformation or rotation.
4. **Landmark tracking:** for facial/hand recordings, integrate thermal-specific landmarks and evaluate tracking confidence, rather than applying a visible-light model without validation.
5. **Independent perfusion reference:** support synchronized LSCI or laser-Doppler traces and agreement/correlation analysis. Without this, outputs remain thermal-pattern descriptors.
6. **Group-level statistics:** only after real participants and study design are available; split by participant, not by frame, to avoid leakage/pseudoreplication.

## Runtime verification note

The local build environment used for this package did not have Streamlit installed and external package-index access was unavailable, so the browser app itself could not be launched here. The Python analysis tests, Plotly figure construction, custom JavaScript syntax, and Python compilation were run locally. Install requirements and execute `streamlit run app.py` on the target machine to perform the browser interaction test.


## v1.2 algorithm/UI refinement

The frame explorer now includes a distinct `Event progression` channel alongside raw temperature, pre-protocol baseline Δ, and dT/dt. An adjustable 1–10 frame event-reference window defaults to a single first-recovery frame. This is essential for baseline + rewarming recordings such as DIRT-SR38: skin can warm after a cooling challenge while remaining below its pre-cooling baseline. The event channel therefore references the first recorded rewarming frames, masks pre-event frames, and computes front-arrival maps relative to that event reference. The selected frame/window is written to the analysis summary and export.

The ROI editor includes a selectable reference frame so anatomical boundaries can be drawn on the clearest image in the sequence. Streamlit Components v2 state values are normalized whether the runtime returns a list or a state object, preventing edited ROIs from being dropped during app reruns. If no ROI is drawn, a separate full-frame analysis region is used without persisting it to the annotation editor.
