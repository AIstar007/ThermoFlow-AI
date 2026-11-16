"""Named polygon ROI validation, rasterization, and time-series extraction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.draw import polygon


def validate_regions(regions: list[dict], image_shape: tuple[int, int], max_regions: int = 8) -> list[dict]:
    h, w = map(int, image_shape)
    if len(regions) > max_regions:
        raise ValueError(f"At most {max_regions} named ROIs are supported in one analysis")
    output: list[dict] = []
    ids: set[str] = set()
    names: set[str] = set()
    for idx, raw in enumerate(regions):
        name = str(raw.get("name", f"ROI {idx + 1}")).strip() or f"ROI {idx + 1}"
        rid = str(raw.get("id", f"roi_{idx + 1}"))
        points = np.asarray(raw.get("points", []), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError(f"{name}: polygon must contain at least three (x, y) points")
        if not np.isfinite(points).all():
            raise ValueError(f"{name}: polygon coordinates must be finite")
        points[:, 0] = np.clip(points[:, 0], 0, w - 1)
        points[:, 1] = np.clip(points[:, 1], 0, h - 1)
        # Shoelace area rejects degenerate polygons.
        x, y = points[:, 0], points[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if area < 1.0:
            raise ValueError(f"{name}: polygon area is too small")
        if rid in ids:
            raise ValueError(f"Duplicate ROI id: {rid}")
        if name.casefold() in names:
            raise ValueError(f"ROI names must be unique (duplicate: {name})")
        ids.add(rid)
        names.add(name.casefold())
        output.append({"id": rid, "name": name, "color": str(raw.get("color", _palette(idx))),
                       "points": points.tolist()})
    return output


def _palette(index: int) -> str:
    palette = ["#4DE2C5", "#FFB45C", "#8FA8FF", "#F17FA8", "#D5B5FF", "#A6D977", "#F27862", "#73D5E8"]
    return palette[index % len(palette)]


def rasterize_regions(regions: list[dict], image_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    h, w = map(int, image_shape)
    masks = {}
    for region in validate_regions(regions, image_shape):
        pts = np.asarray(region["points"], dtype=float)
        rr, cc = polygon(pts[:, 1], pts[:, 0], shape=(h, w))
        mask = np.zeros((h, w), dtype=bool)
        mask[rr, cc] = True
        if not mask.any():
            raise ValueError(f"{region['name']}: polygon did not cover any pixels")
        masks[region["id"]] = mask
    return masks


def roi_metrics_from_result(result, mask: np.ndarray, name: str) -> pd.DataFrame:
    """Extract ROI-specific frame metrics from one full-resolution result.

    Baseline, delta and derivative arrays are shared from a single analysis pass;
    multiple named ROIs therefore do not duplicate the expensive sequence transform.
    """
    roi = np.asarray(mask, dtype=bool)
    if roi.shape != result.baseline.shape or not roi.any():
        raise ValueError("ROI mask must be non-empty and match the thermal frame shape")
    threshold = float(result.summary["applied_change_threshold"])
    rows = []
    for i, t in enumerate(result.metrics["time_s"].to_numpy(dtype=float)):
        values = result._source_frames[i][roi] if hasattr(result, "_source_frames") else None
        if values is None:
            # The analysis result intentionally does not retain a second copy of input frames.
            # Reconstruct a temperature-like trace from baseline + delta.
            values = (result.baseline + result.delta[i])[roi]
        changes = result.delta[i][roi]
        event_changes = result.event_delta[i][roi]
        rates = result.delta_rate[i][roi]
        rows.append({
            "frame": i, "time_s": float(t), "roi": name,
            "mean": float(np.nanmean(values)), "median": float(np.nanmedian(values)),
            "std": float(np.nanstd(values)),
            "mean_delta_from_baseline": float(np.nanmean(changes)),
            "median_delta_from_baseline": float(np.nanmedian(changes)),
            "mean_event_delta": float(np.nanmean(event_changes)) if np.isfinite(event_changes).any() else np.nan,
            "median_event_delta": float(np.nanmedian(event_changes)) if np.isfinite(event_changes).any() else np.nan,
            "mean_dT_dt": float(np.nanmean(rates)) if np.isfinite(rates).any() else np.nan,
            "warming_fraction": float(np.mean(changes >= threshold)),
            "cooling_fraction": float(np.mean(changes <= -threshold)),
            "event_warming_fraction": float(np.mean(event_changes >= threshold)) if np.isfinite(event_changes).any() else np.nan,
            "event_cooling_fraction": float(np.mean(event_changes <= -threshold)) if np.isfinite(event_changes).any() else np.nan,
            "roi_pixels": int(roi.sum()),
        })
    return pd.DataFrame(rows)


def overlap_report(masks: dict[str, np.ndarray]) -> dict:
    if not masks:
        return {"overlap_pixels": 0, "overlap_fraction_of_union": 0.0}
    stack = np.stack(list(masks.values()), axis=0)
    count = stack.sum(axis=0)
    union = count > 0
    overlap = count > 1
    return {"overlap_pixels": int(overlap.sum()),
            "overlap_fraction_of_union": float(overlap.sum() / union.sum()) if union.any() else 0.0}
