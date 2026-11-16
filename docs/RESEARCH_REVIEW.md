# Research review: thermal image sequences and perfusion-related patterns

**Prepared for:** AI/ML Senior Engineer assignment — thermal image processing  
**Review scope:** user-supplied reference papers, peer-reviewed thermal/perfusion studies, a publicly available DIRT sequence dataset, and open-source analysis software.  
**Interpretation rule:** distinguish a directly measured surface-temperature observation from a physiological interpretation or a heat-transfer-model output.

## Executive research conclusion

Dynamic infrared thermography (DIRT) is a more defensible starting point than a single static thermal image. A controlled cooling challenge followed by rewarming gives a temporal response that can be quantified using temperature recovery curves, time constants, area-under-curve features, and spatial hot-spot/arrival maps. Pilot vascular studies and multimodal studies demonstrate that some thermal dynamics covary with laser-based perfusion signals, but this does not make every thermal gradient or moving pattern a direct measurement of blood flow.

The implementation therefore produces two levels of output:

1. **Primary, measured-data outputs:** baseline-relative thermal change, ROI time series, warming/cooling first-arrival maps, and apparent front direction/speed.
2. **Optional model-dependent output:** the heat-balance index in Equation 10 of the 2023 user-supplied paper, only when calibrated Celsius data and ambient temperature are supplied. This remains a model estimate with fixed assumptions; it is not an independent perfusion measurement.

The primary pipeline is deliberately classical and interpretable. Deep learning should be added only when a suitable labelled dataset and a clear prediction target are available.

---

## 1. User-supplied reference papers

### 1.1 Facial Thermal and Blood Perfusion Patterns of Human Emotions: Proof-of-Concept (2023)

**Reference:** Aristizabal-Tique et al., *Journal of Thermal Biology*, 2023, article 103464. DOI: [10.1016/j.jtherbio.2023.103464](https://doi.org/10.1016/j.jtherbio.2023.103464). The accepted manuscript is available at [arXiv:2301.07650](https://arxiv.org/abs/2301.07650).

**What it contributes**

- Establishes a pipeline from thermal facial videos to segmented face maps and region-wise temperature/perfusion-related summaries.
- Uses an FLIR A655SC camera (640×480) and records at 1 frame/s.
- The experimental phases include a 15-minute relaxation/acclimatization period, a baseline recording, and separate emotional-stimulus sequences.
- The analysis uses Otsu thresholding for face segmentation and regions including forehead, eyes, cheeks, nose, and upper lip.
- It calculates ROI average temperature and applies a heat-balance expression attributed to Wu et al. to derive a spatial perfusion-related index.

**Equation 10 used in the paper**

\[
\omega = \frac{\varepsilon\sigma(T_s^4-T_e^4) + A K_f d^{3M-1}\left(\frac{Pr\,g\,\beta}{\nu^2}\right)^M(T_s-T_e)^{M+1} + \frac{k_s(T_a-T_s)}{D} - 4.186}{\alpha c_b(T_a-T_s)}
\]

where the paper defines \(T_s\) as skin-surface temperature, \(T_e\) as ambient temperature, \(T_a\) as arterial/core temperature, and \(\omega\) as a mass-perfusion term in kg/(m²·s). The parameter table sets \(T_a=T_c=312.15\,K\) (39 °C), emissivity 0.98, blood specific heat 3780 J/(kg·K), skin conductivity 0.5 W/(m·K), air conductivity 0.024 W/(m·K), exchange ratio 0.8, and geometric/air constants.

**Critical assessment**

This is explicitly a proof-of-concept with only three participants. The article derives perfusion from surface-temperature images using a physics model and assumed constants; it does not establish an independently validated, patient-specific perfusion meter. The controlled acquisition assumptions (notably fixed environmental conditions and negligible evaporation) matter. The paper's model is included only as an optional comparison because it is directly relevant to the supplied reference, not because its output is known to be clinically accurate for arbitrary data.

**How it informed the implementation**

- Baseline-relative temperature computation and ROI summaries.
- An optional implementation of Equation 10 with Celsius-only input guards.
- Clear model disclaimers and a separate output file for the paper-derived index.

### 1.2 Thermal Infrared Imaging to Evaluate Emotional Competences in Nursing Students: A First Approach through a Case Study (2020)

**Reference:** *Sensors*, 2020;20(9):2502. DOI: [10.3390/s20092502](https://doi.org/10.3390/s20092502). [Open-access full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7248891/).

**What it contributes**

- A practical staged acquisition design: acclimatization, stimulus, and response/recovery phases.
- ROI extraction from facial regions including forehead, nose, and cheeks.
- Emphasizes recording environmental conditions alongside the thermal data.
- Provides a useful example of organizing time-series measurements around an experimental protocol.

**Critical assessment**

It is a first approach/case study with a single participant and focuses on emotional competence in nursing students. It should not be used as evidence that a thermal algorithm can directly quantify blood flow. Its value for this assignment is primarily the experimental workflow and ROI/time-series design.

**How it informed the implementation**

- Explicit baseline period and frame-by-frame ROI summaries.
- Recording FPS, ROI, unit label, and preprocessing settings in the output JSON.

---

## 2. Additional peer-reviewed evidence

### 2.1 Simultaneous IRT and laser-speckle measurement during reactive hyperemia (2023)

**Reference:** Tang et al., “Spectral analysis of laser speckle contrast imaging and infrared thermography to assess skin microvascular reactive hyperemia,” *Skin Research and Technology*, 2023;29:e13308. DOI: [10.1111/srt.13308](https://doi.org/10.1111/srt.13308). [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10234160/).

**Why it is important:** this study recorded hand skin temperature with infrared thermography and blood-flow images with laser speckle contrast imaging (LSCI) in ten healthy volunteers. The protocol included approximately 15 minutes of rest at 24±2 °C, 10 minutes baseline, 5 minutes cuff occlusion, and 10 minutes post-occlusive reactive hyperemia. The authors compared time-frequency signals and found frequency-dependent relationships between IRT and LSCI. They also noted that the signals were non-stationary during cuff inflation/deflation, so the full record should not simply be treated as stationary.

**Implication:** a strong validation experiment for this project would acquire IRT and LSCI simultaneously, align the same ROI, and compare response curves/cross-correlation. LSCI is a reference modality; a thermal-only image sequence cannot provide that reference by itself.

### 2.2 Assessment of chronic limb-threatening ischemia using thermal imaging (2023)

**Reference:** Pakarinen et al., *Journal of Thermal Biology*, 2023;112:103467. DOI: [10.1016/j.jtherbio.2023.103467](https://doi.org/10.1016/j.jtherbio.2023.103467). [Article page](https://www.sciencedirect.com/science/article/pii/S0306456523000086).

**Why it is important:** the pilot study included three healthy young participants, four peripheral artery disease (PAD) patients, and four chronic limb-threatening ischemia (CLTI) patients. It explored thermal modulation and recovery-time analysis alongside clinical reference measurements such as ankle-brachial index (ABI) and toe-brachial index (TBI).

**Key finding:** the authors reported stronger relationships for thermal recovery time constants than for absolute skin temperature or contralateral absolute-temperature differences. They reported correlations of approximately -0.73 with TBI and -0.60 with ABI for recovery time constants, while relationships with absolute temperatures were weak/unclear. The study is a small pilot, not a diagnostic validation trial.

**Implication:** prioritize dynamic recovery descriptors over one-frame hot/cold classification. The project includes an optional exponential rewarming fit to return a descriptive time constant when the experimenter supplies a proper post-cooling recovery segment.

### 2.3 Dynamic infrared thermography systematic review/meta-analysis in flap monitoring (2021)

**Reference:** “Performance of infrared thermography and thermal stress test in perforator mapping and flap monitoring,” Raheman FJ, Rojoa DM, Patel NG. *J Plast Reconstr Aesthet Surg*. 2021;74(9):2013–2025. DOI: [10.1016/j.bjps.2021.03.088](https://doi.org/10.1016/j.bjps.2021.03.088). [PubMed record](https://pubmed.ncbi.nlm.nih.gov/34011474/).

**How to use it:** the review included 18 studies and reported pooled sensitivity/specificity of 99.6%/99.9% for perforator mapping and 89.6%/96.0% for flap monitoring. This supports the maturity of thermography in those specific surgical tasks. Reported diagnostic metrics must not be transferred directly to facial emotion tasks, peripheral artery screening, or this assignment's generic algorithm.

### 2.4 2026 review on dynamic IRT and peripheral microcirculation

**Reference:** “Dynamic infrared thermography for quantitative assessment of peripheral microcirculation: models, metrics, and clinical translation,” *Physiological Measurement*, 2026. DOI: [10.1088/1361-6579/aeb99f](https://doi.org/10.1088/1361-6579/aeb99f). [PubMed record](https://pubmed.ncbi.nlm.nih.gov/42933126/).

This recent review synthesizes 71 studies and modeling papers. The abstract identifies rewarming rate, thermal time constant, temperature-time area under the curve, recovery percentage, and hotspot detection as commonly used quantitative measures. It also highlights the need for standardized protocols and clinical translation. Treat this as a high-level map of current methods, not as a substitute for checking the original validation studies.

---

## 3. Public datasets and what each can/cannot validate

### 3.1 DIRT-SR38 — primary recommendation

- **Dataset:** [Zenodo DOI 10.5281/zenodo.19241293](https://doi.org/10.5281/zenodo.19241293).
- **Paper:** Clarys et al., “DIRT-SR38: A Real Paired Cross-Sensor Thermal Sequence Dataset for Super-Resolution in Clinical Dynamic Infrared Thermography,” CVPR Workshops 2026.
- 38 healthy adult female participants, each with a 30-second steady-state baseline and four-minute dynamic rewarming sequence.
- 810 aligned LR/HR frame pairs per subject (30,780 in total); LR images are 108×108 and HR 216×216.
- The released data are intensity-normalized PNGs. They preserve relative spatial thermal patterns but are not calibrated degrees Celsius.
- The dataset is for research in super-resolution, cross-sensor modeling, and DIRT analysis. The paper explicitly says it is not intended for clinical diagnosis or direct patient care.
- The Zenodo record lists a 3.6 GB archive. The license field is not populated on the record page reviewed; inspect current rights/terms before reuse or redistribution.

**Fit to this project:** the best real sequence dataset identified for temporal thermal analysis. Use one subject, one modality (LR or HR), and the matching baseline + recovery frames. The paper's LR stream was released at 3 Hz. For the 150×150 mm ROI, the approximate nominal pitches are 1.389 mm/px for 108×108 LR and 0.694 mm/px for 216×216 HR.

**Limit:** it has no co-recorded LSCI/LDF perfusion ground truth in the released thermal PNG set and no absolute °C values. It supports evaluation of relative rewarming patterns, not direct blood-flow calibration.

### 3.2 FDA Facial and Oral Temperature Dataset

- [FDA dataset portal](https://www.fda.gov/medical-devices/thermographic-systems/facial-and-oral-temperature-dataset).
- [PhysioNet record](https://physionet.org/content/facial-and-oral-temperature-data/1.0/).
- [Companion GitHub tables](https://github.com/dbp-osel/Facial-and-oral-temperature-data).

Useful for static facial thermal and oral-temperature comparison and for checking data-ingestion/reporting. It is not a dynamic perfusion ground-truth sequence dataset, so it cannot validate a propagation velocity or thermal recovery time constant.

### 3.3 Dynamic FLIR sepsis dataset

- [DataverseLV DOI 10.71782/DATA/DPLSCG](https://doi.org/10.71782/DATA/DPLSCG).

The published record describes paired baseline/cooled thermal measurement matrices for 21 sepsis patients using a FLIR A655sc. It can be useful for before/after spatial thermal-change testing, but the available description is not a full frame-resolved rewarming video sequence. Check the released file inventory before treating it as sequential data.

### 3.4 Practical validation dataset gap

The strongest public DIRT sequence data identified here are normalized PNGs without °C calibration or a direct perfusion reference. A full validation study should therefore add a prospectively acquired dataset with:

- radiometric thermal sequences and camera metadata;
- standardized ambient conditions and acclimatization;
- documented cooling/stimulation and recovery timestamps;
- synchronized LSCI, laser Doppler flowmetry, or another independent perfusion reference;
- repeated participants and test-retest recordings;
- participant-level splits for model development (never randomly split correlated frames across train/test).

---

## 4. Open-source projects and GitHub repositories

### 4.1 TPakar/IRlab — strongest software reference

[GitHub: TPakar/IRlab](https://github.com/TPakar/IRlab)  
Paper: Pakarinen et al., “IRlab – Platform for thermal video analysis in evaluation of peripheral thermal behavior and blood perfusion,” *Informatics in Medicine Unlocked*, 2022;30:100940. DOI: [10.1016/j.imu.2022.100940](https://doi.org/10.1016/j.imu.2022.100940).

The paper describes a MATLAB research platform with ROI tools, thermal modulation, median/maximum/minimum/standard-deviation measurements, subtraction/gamma maps, spectral-band power, spectral flow, wavelet tools, automatic bilateral registration, and thermal video playback. The public repository includes code for data import, ROI drawing, registration, artifact handling, filtering, wavelets, and spectral-flow processing. It is a research tool and a useful reference for workflow choices.

**Use in this project:** method comparison and requirements reference only. The Python project is written independently and does not copy the MATLAB source. IRlab also implements frequency-band filtering and spectral-flow tools; those are not enabled in this first Python baseline because low-rate DIRT recordings and stimulus/recovery transitions require careful frequency/Nyquist and stationarity checks before a spectral output is labelled as perfusion.

### 4.2 KISMED-TUDa/ISBI_GSR_Thermal_Videos — facial landmark stabilization

[GitHub: KISMED-TUDa/ISBI_GSR_Thermal_Videos](https://github.com/KISMED-TUDa/ISBI_GSR_Thermal_Videos).

The repository accompanies a 2026 IEEE ISBI paper on thermal-aware YOLO-Pose facial keypoints and thermal video stabilization for thermal/GSR analysis. It describes 54 facial landmarks, a pipeline that applies the same geometric transform to grayscale and temperature arrays, and example data formats. This is especially relevant when the target ROI is a moving face. It is less relevant to fixed-camera abdominal/limb DIRT sequences.

### 4.3 dbp-osel/Facial-and-oral-temperature-data — FDA companion data

[GitHub: dbp-osel/Facial-and-oral-temperature-data](https://github.com/dbp-osel/Facial-and-oral-temperature-data).

Contains data tables and companion material for facial/oral-temperature work. Useful for static data checks; not a temporal perfusion sequence implementation.

### 4.4 joydelavictoria/thermography-perfusion-demo — useful synthetic/statistical example

[GitHub: joydelavictoria/thermography-perfusion-demo](https://github.com/joydelavictoria/thermography-perfusion-demo).

The README clearly identifies all data as synthetic. Its value is the statistical discussion: frames from one animal/participant are not independent subjects, so treating every frame as an independent sample causes pseudo-replication. It also demonstrates fixed-bin thermal descriptors and participant-level permutation tests. This repository should be treated as an educational simulation, not clinical validation or a production algorithm.

---

## 5. Real-world applications and their relevance

| Application | Typical thermal experiment | Relevance to this assignment | Evidence caveat |
|---|---|---|---|
| Peripheral vascular assessment / PAD / CLTI | Controlled thermal modulation, then measure regional rewarming/recovery | Strong fit for time constants, ROI asymmetry, and spatial recovery patterns | Small clinical pilot studies; not a standalone diagnostic method |
| Raynaud's phenomenon / digital microcirculation | Cold challenge or reactive hyperemia with fingertip temperature recovery | Useful for recovery curves and comparison with laser Doppler/LSCI | Skin temperature depends on external conditions and vascular tone; validation requires reference measurements |
| Reconstructive surgery / perforator mapping / flap monitoring | DIRT cooling followed by rewarming and hot-spot mapping | Strong real-world fit for dynamic ROI maps and recovery metrics | Clinical performance in flap tasks does not establish accuracy for other conditions |
| Facial emotion/affect research | Thermal video during baseline and emotional stimuli | Fits facial ROI and motion-stabilization methods from the supplied papers | Emotional thermal shifts are not the same as clinical perfusion |
| Sepsis/critical care | Paired skin thermal states or thermal stress response | Possible exploratory monitoring of peripheral thermal heterogeneity | Requires independent perfusion/clinical labels and careful acquisition control |

---

## 6. Why the implementation uses arrival maps instead of naive optical flow

The assignment asks for direction and propagation of thermal patterns. Conventional optical flow estimates image texture displacement, not the motion of a temperature response. If the image is nearly static while a region gradually warms, optical flow can be weak or misleading.

This project instead measures when each pixel first crosses a defined temperature-change threshold. The resulting arrival-time surface encodes the order of thermal activation. Its spatial gradient estimates the normal direction and speed of that apparent threshold front:

\[
\mathbf{n}(x,y)=\frac{\nabla t_a(x,y)}{\|\nabla t_a(x,y)\|},\qquad
v_n(x,y)=\frac{1}{\|\nabla t_a(x,y)\|}.
\]

This is interpretable and can be validated with a synthetic front of known speed. It is still a description of a temperature-threshold front, not blood movement. It also depends on the threshold, sample rate, smoothing, and frame stability.

---

## 7. Recommended next research/validation stage

1. Select two or more DIRT-SR38 subjects; process LR and HR separately and keep one participant entirely in one split.
2. Use the 30-second baseline as the reference period and annotate the four-minute rewarming segment.
3. Inspect frame-to-frame median intensity for non-uniformity-correction jumps or global drift. Do not treat camera corrections as tissue responses.
4. Compare ROI mean/median recovery, recovery slope, time constant, area-under-curve, and first-arrival spatial maps.
5. Run sensitivity analysis across thresholds and smoothing settings; report direction and velocity stability.
6. Acquire or obtain paired LSCI/LDF/clinical reference measurements before using the word “perfusion estimate” as an evaluated output.
7. Validate across participants with subject-level holdout and report uncertainty, not just a visually plausible heatmap.

## References and links

1. Aristizabal-Tique VH et al. Facial Thermal and Blood Perfusion Patterns of Human Emotions: Proof-of-Concept. *J Therm Biol*. 2023; article 103464. [DOI](https://doi.org/10.1016/j.jtherbio.2023.103464) · [accepted manuscript](https://arxiv.org/abs/2301.07650).
2. Thermal Infrared Imaging to Evaluate Emotional Competences in Nursing Students: A First Approach through a Case Study. *Sensors*. 2020;20(9):2502. [DOI](https://doi.org/10.3390/s20092502) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7248891/).
3. Tang Y, Xu F, Lei P, Li G, Tan Z. Spectral analysis of laser speckle contrast imaging and infrared thermography to assess skin microvascular reactive hyperemia. *Skin Res Technol*. 2023;29:e13308. [DOI](https://doi.org/10.1111/srt.13308) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10234160/).
4. Pakarinen T, Joutsen A, Oksala N, Vehkaoja A. Assessment of chronic limb threatening ischemia using thermal imaging. *J Therm Biol*. 2023;112:103467. [DOI](https://doi.org/10.1016/j.jtherbio.2023.103467).
5. Pakarinen T et al. IRlab – Platform for thermal video analysis in evaluation of peripheral thermal behavior and blood perfusion. *Inform Med Unlocked*. 2022;30:100940. [DOI](https://doi.org/10.1016/j.imu.2022.100940) · [GitHub](https://github.com/TPakar/IRlab).
6. Clarys W et al. DIRT-SR38: A Real Paired Cross-Sensor Thermal Sequence Dataset for Super-Resolution in Clinical Dynamic Infrared Thermography. CVPR Workshops 2026. [paper](https://openaccess.thecvf.com/content/CVPR2026W/PBVS/papers/Clarys_DIRT-SR38_A_Real_Paired_Cross-Sensor_Thermal_Sequence_Dataset_for_Super-Resolution_CVPRW_2026_paper.pdf) · [dataset](https://doi.org/10.5281/zenodo.19241293).
7. “Dynamic infrared thermography for quantitative assessment of peripheral microcirculation: models, metrics, and clinical translation.” *Physiol Meas*. 2026. [DOI](https://doi.org/10.1088/1361-6579/aeb99f).
8. KISMED-TUDa. Thermal-aware YOLO-Pose landmark detection and stabilization repository. [GitHub](https://github.com/KISMED-TUDa/ISBI_GSR_Thermal_Videos).
9. FDA Facial and Oral Temperature Dataset. [FDA](https://www.fda.gov/medical-devices/thermographic-systems/facial-and-oral-temperature-dataset) · [PhysioNet](https://physionet.org/content/facial-and-oral-temperature-data/1.0/).
10. Dynamic FLIR Sepsis dataset. [DataverseLV](https://doi.org/10.71782/DATA/DPLSCG).


## Additional integration literature and platform resources (reviewed for v1.3–v1.4)

### Thermal and independent perfusion reference

- Tang et al. (2023), *Spectral Analysis of Laser Speckle Contrast Imaging and Infrared Thermography to Assess Skin Microvascular Reactive Hyperemia*, [PMC10234160](https://pmc.ncbi.nlm.nih.gov/articles/PMC10234160/). A small simultaneous LSCI/IRT study in healthy volunteers. It motivates paired reference measurements, frequency-aware signal analysis and camera-sensitivity checks; it does not make thermography a substitute for LSCI.
- Muller et al. (2024), *Combining thermal imaging and machine learning to noninvasively characterize palm perfusion during local blood pressure changes*, [DOI:10.1016/j.bspc.2024.106109](https://doi.org/10.1016/j.bspc.2024.106109). Relevant to regional thermal features and machine-learning exploration; the sample and experimental setting limit generalization.
- 2026 wearable thermal-gradient sensing work, *Deep learning-integrated multilayer thermal gradient sensing platform for real-time blood flow monitoring*, [PMC12880533](https://pmc.ncbi.nlm.nih.gov/articles/PMC12880533/). This uses purpose-built multilayer thermal hardware and PPG. It is not equivalent to passive infrared video and should not be copied as a model for this app without the required hardware and paired training data.

### Camera SDK and acquisition resources

- FLIR Science File SDK documentation: [FLIR support](https://flir.custhelp.com/app/answers/detail/a_id/6622/related/1). It is for decoding supported recorded scientific files; camera control/live acquisition is a separate Science Camera SDK pathway.
- InfraTec SDK overview: [IRBACS / IRBGRAB](https://www.infratec-infrared.com/thermography/thermographic-software/sdk/). IRBACS supports stored IRB workflows; IRBGRAB provides online acquisition, subject to camera-specific SDK availability/licensing.
- Optris OTC SDK: [official SDK downloads](https://optris.github.io/otcsdk_downloads/). Published SDK materials describe Python 3 and supported USB/Ethernet workflows for compatible PI/Xi cameras.
- Seek Thermal Python bindings: [seekcamera-python](https://github.com/seekthermal/seekcamera-python). Official bindings require the corresponding Seek SDK 4.x and compatible physical OEM hardware; not all consumer Seek devices should be assumed supported.
- GenICam/GenTL acquisition via [Harvester](https://harvesters.readthedocs.io/). A GenTL producer may expose image payloads; radiometric decoding/calibration may remain vendor-specific.

### Modelling and reproducibility resources

- The v1.3 Pennes module uses the classical bioheat-equation framework as a simplified numerical teaching model. It is not a reconstruction of a particular published tissue model, and it is not an inverse perfusion estimator.
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking/) is used as an optional parameter/metric/artifact log.
- [DVC documentation](https://dvc.org/doc) provides a recommended approach for versioning large research datasets and reproducible pipelines.
- [Streamlit Components v2](https://docs.streamlit.io/develop/concepts/custom-components/overview) supports custom bidirectional UI components; [Plotly animations](https://plotly.com/python/animations/) provide client-side animated charts.

These are integration resources and design precedents, not claims that all vendor drivers or hardware integrations are bundled or tested. Camera profiles are selectable; direct native decoding/live acquisition remains contingent on each model's SDK and a matching adapter plugin.


## Additional v1.4 integration references (checked 2026-11-16)

- FLIR, [Science Camera SDK download and product description](https://flir.custhelp.com/app/answers/detail/a_id/6623/~/science-camera-sdk---download) (camera control/acquisition; Python bindings; commercial license) and [Science File SDK](https://flir.custhelp.com/app/answers/detail/a_id/6622/related/1) (post-processing of radiometric files, not camera control). FLIR provides a [camera-to-SDK chooser](https://flir.custhelp.com/app/answers/detail/a_id/3000/related/1), since SDK choice depends on the camera family.
- InfraTec, [SDK integration for online and offline applications](https://www.infratec.eu/thermography/thermographic-software/sdk/): IRBGRAB for online acquisition; IRBACS for stored IRB reading and corrected-temperature conversion.
- Optris, [OTC SDK](https://optris.com/software/otc-sdk/) and [feature matrix](https://optris.github.io/otcsdk_downloads/features.html): Python API, USB/Ethernet, thermal data, timestamps/calibration metadata and multi-camera support for supported PI/Xi devices.
- Seek Thermal, [official seekcamera-python bindings](https://github.com/seekthermal/seekcamera-python): requires Seek SDK 4.x and compatible attached OEM Core hardware.
- HIKMICRO, [industrial thermal camera interfaces](https://www.hikmicrotech.com/en/industrial-products/furnace-cameras/): certain models publish 16-/32-bit thermal data matrices and model-specific network/SDK pathways; not a universal capability across every device.
- Testo, [official GitHub organization](https://github.com/orgs/testo/repositories): public model-specific API examples exist; compatibility must be determined per model.
- Lab Streaming Layer, [time synchronization documentation](https://labstreaminglayer.readthedocs.io/info/time_synchronization.html): per-sample/chunk timestamps and periodic clock-offset estimates; XDF importers can synchronize from the recorded clock-offset history.
- PyXDF, [PyPI](https://pypi.org/project/pyxdf/) and [v1.18 changelog](https://github.com/xdf-modules/pyxdf/blob/main/CHANGELOG.md): XDF clock sync, clock reset handling, and optional timestamp de-jitter.
- pylsl, [PyPI](https://pypi.org/project/pylsl/): Python LSL API; underlying `liblsl` availability depends on platform/wheel.
- Streamlit, [Components v2 bidirectional communication](https://docs.streamlit.io/develop/concepts/custom-components/components-v2/communicate): Python-to-browser data and callback state exchange.

These references support the integration architecture, not an assertion that all devices have been physically connected or validated in this build.


## v1.4 official camera and stream integration research

The v1.4 camera catalogue is based on current official vendor material rather than an assumption that all thermal cameras share one driver:

- **FLIR:** the Science Camera SDK is a Python-capable control/acquisition API and is separate from the Science File SDK, which decodes stored radiometric recordings. SDK choice depends on camera model and licensing. See [Science Camera SDK](https://flir.custhelp.com/app/answers/detail/a_id/6623/~/science-camera-sdk---download), [Science File SDK](https://flir.custhelp.com/app/answers/detail/a_id/6622/related/1), and [FLIR SDK chooser](https://flir.custhelp.com/app/answers/detail/a_id/3000/related/1).
- **Optris:** the OTC SDK documents Python 3, USB/Ethernet, all current PI/Xi cameras, temperature conversion, timestamp/calibration metadata, connection monitoring and multi-camera support. A local adapter still needs to map the vendor SDK into this project's scalar-frame/timestamp contract. See [OTC SDK docs](https://optris.github.io/otcsdk_downloads/) and [features](https://optris.github.io/otcsdk_downloads/features.html).
- **Seek Thermal:** the official Python bindings require Seek SDK 4.x and a physically attached compatible OEM core; they are not a general driver for every retail Seek camera. See [seekcamera-python](https://github.com/seekthermal/seekcamera-python).
- **InfraTec:** IRBGRAB is the online acquisition SDK, whereas IRBACS handles stored IRB data and corrected-temperature conversion. Live and offline SDK capabilities are distinct and may have camera-specific licensing. See [InfraTec SDK](https://www.infratec-infrared.com/thermography/thermographic-software/sdk/).
- **HIKMICRO, Testo, Fluke and InfiRay/Guide:** interface and radiometric payload support vary by exact model and firmware; the project advertises only model-specific adapter paths for these vendors. Generic palette video is not treated as radiometric data.

The data transport layer supports LabRecorder XDF via [pyxdf](https://pypi.org/project/pyxdf/) and live numeric streams via [pylsl](https://pypi.org/project/pylsl/) / [Lab Streaming Layer](https://labstreaminglayer.readthedocs.io/info/time_synchronization.html). Clock synchronization, timestamp de-jitter and cross-device camera alignment are treated as separate operations.
