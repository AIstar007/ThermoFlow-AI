import json
import zipfile

import numpy as np
from PIL import Image

from thermoprop.io import load_array, load_image_zip
from thermoprop.pipeline import run_pipeline
from thermoprop.synthetic import make_propagation_sequence


def test_npz_reader_prefers_temperature_key(tmp_path):
    path = tmp_path / "sample.npz"
    expected = np.ones((5, 8, 9), dtype=np.float32) * 30
    np.savez(path, temperatures=expected, array1=np.zeros((5, 8, 9)))
    actual = load_array(path)
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected)


def test_zip_reader_orders_image_frames_naturally(tmp_path):
    path = tmp_path / "sequence.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in [("frame10.png", 10), ("frame2.png", 2), ("frame1.png", 1)]:
            image_path = tmp_path / name
            Image.fromarray(np.full((8, 9), value, dtype=np.uint8)).save(image_path)
            archive.write(image_path, arcname=name)
    frames = load_image_zip(path.read_bytes())
    assert frames.shape == (3, 8, 9)
    assert frames[:, 0, 0].tolist() == [1.0, 2.0, 10.0]


def test_pipeline_exports_reproducible_artifacts(tmp_path):
    frames, _ = make_propagation_sequence(n_frames=40, seed=14)
    output = tmp_path / "run"
    result = run_pipeline(frames, output_dir=output, fps=2, baseline_frames=4,
                          threshold=0.3, units="synthetic", sequence_name="test")
    assert (output / "analysis_summary.png").is_file()
    assert (output / "frame_metrics.csv").is_file()
    assert (output / "thermal_analysis_arrays.npz").is_file()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["sequence_name"] == "test"
    assert result.summary["warming_propagation"]["direction"] == "right"
