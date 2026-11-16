"""Optional thermal-domain YOLO segmentation adapter for ROI suggestions.

Requires ``pip install -e '.[ai]'`` and a user-supplied, trusted YOLO-seg model
trained for the relevant thermal camera domain. Generic RGB weights are not
assumed to be valid for thermal imagery.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


def _to_model_image(frame: np.ndarray, mode: str = "percentile", fixed_min: float | None = None,
                    fixed_max: float | None = None) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("Selected frame contains no finite pixel data")
    if mode == "percentile":
        lo, hi = np.percentile(arr[finite], [1, 99])
    elif mode == "fixed":
        if fixed_min is None or fixed_max is None or fixed_max <= fixed_min:
            raise ValueError("Fixed model normalization needs maximum > minimum")
        lo, hi = float(fixed_min), float(fixed_max)
    else:
        raise ValueError("Normalization mode must be 'percentile' or 'fixed'")
    if hi <= lo:
        hi = lo + 1.0
    gray = np.clip((np.nan_to_num(arr, nan=lo) - lo) / (hi - lo), 0, 1)
    gray = np.rint(gray * 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def suggest_rois_yolo_seg(
    frame: np.ndarray,
    model_bytes: bytes,
    filename: str = "thermal_segmentation.pt",
    confidence: float = 0.35,
    image_size: int = 640,
    normalization: str = "percentile",
    fixed_min: float | None = None,
    fixed_max: float | None = None,
    max_regions: int = 8,
) -> list[dict]:
    if not model_bytes:
        raise ValueError("Upload a trusted YOLO segmentation model first")
    if not 0 < float(confidence) < 1:
        raise ValueError("Confidence must be between 0 and 1")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("AI segmentation dependencies are missing. Install with: pip install -e '.[ai]'") from exc

    image = _to_model_image(frame, normalization, fixed_min, fixed_max)
    suffix = Path(filename).suffix or ".pt"
    if suffix.lower() not in {".pt", ".onnx"}:
        raise ValueError("Upload a YOLO segmentation model with .pt or .onnx extension")
    # Only temporary copy of the uploaded model; it is not retained after inference.
    with tempfile.TemporaryDirectory(prefix="thermoprop_model_") as temp_dir:
        model_path = Path(temp_dir) / ("trusted_model" + suffix)
        model_path.write_bytes(model_bytes)
        model = YOLO(str(model_path), task="segment")
        predictions = model.predict(source=image, conf=float(confidence), imgsz=int(image_size),
                                    verbose=False, retina_masks=True)
    if not predictions:
        return []
    prediction = predictions[0]
    if prediction.masks is None or prediction.masks.xy is None:
        raise ValueError("The uploaded model returned no segmentation masks. It must be a YOLO segmentation model, not a detection-only model.")
    names = getattr(prediction, "names", {}) or {}
    boxes = getattr(prediction, "boxes", None)
    confs = boxes.conf.cpu().numpy().tolist() if boxes is not None and boxes.conf is not None else []
    classes = boxes.cls.cpu().numpy().astype(int).tolist() if boxes is not None and boxes.cls is not None else []
    palette = ["#4DE2C5", "#FFB45C", "#8FA8FF", "#F17FA8", "#D5B5FF", "#A6D977", "#F27862", "#73D5E8"]
    regions = []
    for idx, polygon in enumerate(prediction.masks.xy[:max_regions]):
        points = np.asarray(polygon, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            continue
        cls_id = classes[idx] if idx < len(classes) else -1
        label = (names.get(cls_id, f"class_{cls_id}") if isinstance(names, dict)
                 else names[cls_id] if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names)
                 else f"class_{cls_id}")
        conf = float(confs[idx]) if idx < len(confs) else float("nan")
        label_safe = str(label).replace("/", "-").strip() or "region"
        regions.append({"id": f"ai_{idx+1}", "name": f"AI {label_safe} {idx+1} ({conf:.2f})",
                        "color": palette[idx % len(palette)], "points": points.tolist()})
    return regions
