from __future__ import annotations

import json
from pathlib import Path

from thermoprop.pipeline import run_pipeline
from thermoprop.synthetic import make_propagation_sequence

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "synthetic_demo"
OUT.mkdir(parents=True, exist_ok=True)

all_results = {}
for name, direction, seed in [
    ("sequence_01_warming_right", "right", 21),
    ("sequence_02_cooling_left", "left", 22),
]:
    frames, truth = make_propagation_sequence(direction=direction, seed=seed)
    seq_dir = OUT / name
    seq_dir.mkdir(parents=True, exist_ok=True)
    npy_path = seq_dir / "input_frames.npy"
    import numpy as np
    np.save(npy_path, frames)
    (seq_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    result = run_pipeline(
        frames,
        output_dir=seq_dir,
        fps=truth["fps"],
        baseline_frames=4,
        threshold=0.30,
        units="°C (synthetic)",
        sequence_name=name,
        extra_metadata={"ground_truth": truth, "data_origin": "synthetic; no patient data"},
    )
    all_results[name] = {"ground_truth": truth, "estimated_summary": result.summary}

(OUT / "demo_results.json").write_text(json.dumps(all_results, indent=2, allow_nan=True), encoding="utf-8")
print(json.dumps({
    name: {
        "expected_direction": data["ground_truth"]["expected_direction"],
        "warming_estimate": data["estimated_summary"]["warming_propagation"],
        "cooling_estimate": data["estimated_summary"]["cooling_propagation"],
        "warming_fraction": data["estimated_summary"]["warming_event_fraction"],
        "cooling_fraction": data["estimated_summary"]["cooling_event_fraction"],
    } for name, data in all_results.items()
}, indent=2))
