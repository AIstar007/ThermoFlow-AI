from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_sequence, npz_keys
from .pipeline import run_pipeline


def _roi(value: str):
    try:
        parts = tuple(int(v.strip()) for v in value.split(","))
        if len(parts) != 4:
            raise ValueError
        return parts
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be x0,y0,x1,y1") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze thermal sequence temperature changes and apparent propagation")
    parser.add_argument("input", help="Path to .npy, .npz, .tif/.tiff, .csv, ZIP of ordered images, or image folder")
    parser.add_argument("--output", default="results/run", help="Output directory")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame rate (ignored when timestamps are supplied by future integrations)")
    parser.add_argument("--baseline-frames", type=int, default=5, help="Number of initial baseline frames")
    parser.add_argument("--threshold", type=float, default=0.25, help="Minimum change threshold in the input units")
    parser.add_argument("--units", default="relative intensity", choices=["C", "°C", "relative intensity", "camera counts"], help="Label only; does not calibrate the input")
    parser.add_argument("--npz-key", default=None, help="Array key to read from an NPZ")
    parser.add_argument("--roi", type=_roi, default=None, help="ROI rectangle x0,y0,x1,y1; default is full frame")
    parser.add_argument("--pixel-size-mm", type=float, default=None, help="Pixel pitch in mm/pixel; enables speed in mm/s")
    parser.add_argument("--register", action="store_true", help="Enable conservative translational frame registration")
    parser.add_argument("--spatial-sigma", type=float, default=0.6, help="Gaussian denoising sigma in pixels; 0 disables")
    parser.add_argument("--name", default=None, help="Sequence label")
    parser.add_argument("--paper-model-index", action="store_true", help="Compute the optional Eq. 10 heat-balance index (requires calibrated Celsius input)")
    parser.add_argument("--ambient-temp-c", type=float, default=None, help="Ambient temperature in °C for the optional paper model")
    parser.add_argument("--arterial-temp-c", type=float, default=39.0, help="Arterial/core temperature in °C (paper default 39.0)")
    parser.add_argument("--recovery-start-frame", type=int, default=None, help="Optional first frame of the post-cooling rewarming segment; adds a descriptive exponential time constant")
    args = parser.parse_args(argv)

    path = Path(args.input)
    if path.suffix.lower() == ".npz" and args.npz_key is None:
        keys = npz_keys(path)
    else:
        keys = None
    frames = load_sequence(path, npz_key=args.npz_key)
    name = args.name or path.stem
    if args.paper_model_index and args.ambient_temp_c is None:
        parser.error("--paper-model-index requires --ambient-temp-c")
    result = run_pipeline(
        frames, output_dir=args.output, fps=args.fps, baseline_frames=args.baseline_frames,
        threshold=args.threshold, roi_xyxy=args.roi, units=args.units, register=args.register,
        spatial_sigma=args.spatial_sigma, pixel_size_mm=args.pixel_size_mm, sequence_name=name,
        extra_metadata={"source_path": str(path), "npz_keys": keys},
        paper_model_ambient_c=args.ambient_temp_c if args.paper_model_index else None,
        paper_model_arterial_c=args.arterial_temp_c,
        recovery_start_frame=args.recovery_start_frame,
    )
    print(json.dumps(result.summary, indent=2, allow_nan=True))
    print(f"\nSaved analysis to: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
