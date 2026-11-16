"""End-to-end analysis and artifact export."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .analysis import analyze_sequence, rectangular_roi
from .bioheat import wu_style_model_index
from .preprocess import preprocess_sequence
from .recovery import fit_rewarming_time_constant
from .visualization import create_summary_figure, write_json


def run_pipeline(
    frames: np.ndarray,
    output_dir: str | Path,
    fps: float = 1.0,
    baseline_frames: int = 5,
    threshold: float = 0.25,
    roi_xyxy: tuple[int, int, int, int] | None = None,
    units: str = "relative intensity",
    register: bool = False,
    spatial_sigma: float = 0.6,
    pixel_size_mm: float | None = None,
    sequence_name: str = "thermal_sequence",
    extra_metadata: dict | None = None,
    paper_model_ambient_c: float | None = None,
    paper_model_arterial_c: float = 39.0,
    recovery_start_frame: int | None = None,
):
    if paper_model_ambient_c is not None and units not in {"C", "°C"}:
        raise ValueError("The paper-based heat-balance index requires calibrated Celsius input; relative PNG intensities cannot be used.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if roi_xyxy is None:
        roi = np.ones(frames.shape[1:], dtype=bool)
    else:
        roi = rectangular_roi(frames.shape[1:], *roi_xyxy)
    processed, prep_metadata = preprocess_sequence(frames, spatial_sigma=spatial_sigma, register=register,
                                                   registration_reference_frames=baseline_frames)
    result = analyze_sequence(processed, fps=fps, roi_mask=roi, baseline_frames=baseline_frames,
                              change_threshold=threshold, pixel_size_mm=pixel_size_mm)
    result.summary["sequence_name"] = sequence_name
    result.summary["units"] = units
    result.summary["preprocessing"] = prep_metadata
    result.summary["metadata"] = extra_metadata or {}
    result.summary["roi_bounds_xyxy"] = list(roi_xyxy) if roi_xyxy else [0, 0, frames.shape[2], frames.shape[1]]
    if recovery_start_frame is not None:
        result.summary["rewarming_time_constant_fit"] = fit_rewarming_time_constant(
            result.metrics["time_s"].to_numpy(), result.metrics["median"].to_numpy(),
            start_index=int(recovery_start_frame),
        )

    result.metrics.to_csv(out / "frame_metrics.csv", index=False)
    np.savez_compressed(
        out / "thermal_analysis_arrays.npz",
        baseline=result.baseline,
        delta=result.delta,
        delta_rate=result.delta_rate,
        roi_mask=result.roi_mask.astype(np.uint8),
        warming_first_arrival_s=result.warming_arrival_s,
        cooling_first_arrival_s=result.cooling_arrival_s,
    )
    create_summary_figure(result, processed, out / "analysis_summary.png", unit_label=units)
    if paper_model_ambient_c is not None:
        model_index = wu_style_model_index(result.baseline, ambient_temperature_c=paper_model_ambient_c,
                                           arterial_temperature_c=paper_model_arterial_c)
        model_index[~result.roi_mask] = np.nan
        np.save(out / "paper_model_perfusion_index_kg_m2_s.npy", model_index)
        result.summary["paper_model_index"] = {
            "computed": True,
            "model": "Aristizabal-Tique et al. (2023), Eq. 10 / Wu-style heat-balance model",
            "ambient_temperature_c": float(paper_model_ambient_c),
            "arterial_temperature_c": float(paper_model_arterial_c),
            "finite_pixel_fraction_in_roi": float(np.isfinite(model_index[result.roi_mask]).mean()),
            "warning": "Model-dependent index with fixed assumptions; not directly measured or clinically validated perfusion.",
        }
        np.savez_compressed(out / "thermal_analysis_arrays.npz",
                            baseline=result.baseline, delta=result.delta, delta_rate=result.delta_rate,
                            roi_mask=result.roi_mask.astype(np.uint8),
                            warming_first_arrival_s=result.warming_arrival_s,
                            cooling_first_arrival_s=result.cooling_arrival_s,
                            paper_model_perfusion_index=model_index)
    write_json(result.summary, out / "summary.json")
    return result
