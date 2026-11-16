"""Conservative thermal-sequence preprocessing and optional rigid registration."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, shift
from skimage.registration import phase_cross_correlation


def preprocess_sequence(
    frames: np.ndarray,
    spatial_sigma: float = 0.6,
    register: bool = False,
    registration_reference_frames: int = 5,
    registration_mode: str = "phase_translation",
) -> tuple[np.ndarray, dict]:
    """Filter small pixel noise and optionally correct translational motion.

    Registration is deliberately opt-in: an actual thermal response can itself alter
    image texture, and a wrong transform can create artificial temperature gradients.
    """
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] < 2:
        raise ValueError("frames must have shape (n_frames, height, width), with n_frames >= 2")
    if not np.isfinite(arr).any():
        raise ValueError("Sequence contains no finite values")
    if not np.isfinite(arr).all():
        # Fill isolated invalid pixels per frame with that frame's finite median.
        for i in range(arr.shape[0]):
            finite = np.isfinite(arr[i])
            fill = float(np.nanmedian(arr[i])) if finite.any() else np.nan
            if not np.isfinite(fill):
                raise ValueError(f"Frame {i} has no finite measurements")
            arr[i] = np.where(finite, arr[i], fill)

    if spatial_sigma > 0:
        arr = np.stack([gaussian_filter(frame, sigma=spatial_sigma, mode="nearest") for frame in arr])

    shifts: list[list[float]] = [[0.0, 0.0] for _ in range(len(arr))]
    transforms: list[list[float]] = []
    if registration_mode not in {"phase_translation", "affine_ecc"}:
        raise ValueError("registration_mode must be 'phase_translation' or 'affine_ecc'")
    if register:
        nref = max(1, min(registration_reference_frames, len(arr)))
        reference = np.median(arr[:nref], axis=0).astype(np.float32)
        aligned = np.empty_like(arr)
        if registration_mode == "affine_ecc":
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("Affine ECC registration requires OpenCV. Install with: pip install -e '.[vision]'") from exc
            h, w = reference.shape
            template = cv2.normalize(reference, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
            for idx, frame in enumerate(arr):
                moving = cv2.normalize(frame, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
                warp = np.eye(2, 3, dtype=np.float32)
                try:
                    _, warp = cv2.findTransformECC(template, moving, warp, cv2.MOTION_AFFINE, criteria,
                                                   None, 5)
                    # Guard against unstable scale/rotation/shear or large translation.
                    linear = warp[:, :2]
                    det = float(np.linalg.det(linear))
                    scale = float(np.sqrt(abs(det)))
                    tx, ty = float(warp[0, 2]), float(warp[1, 2])
                    if not (0.85 <= scale <= 1.18 and abs(tx) <= 0.15*w and abs(ty) <= 0.15*h and 0.70 <= abs(det) <= 1.40):
                        warp = np.eye(2, 3, dtype=np.float32)
                    aligned[idx] = cv2.warpAffine(frame, warp, (w, h),
                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_REPLICATE)
                    shifts[idx] = [float(warp[1, 2]), float(warp[0, 2])]
                    transforms.append(warp.reshape(-1).astype(float).tolist())
                except cv2.error:
                    aligned[idx] = frame
                    transforms.append(np.eye(2, 3, dtype=np.float32).reshape(-1).astype(float).tolist())
        else:
            for idx, frame in enumerate(arr):
                delta, _, _ = phase_cross_correlation(reference, frame, upsample_factor=10, normalization=None)
                # Guard against implausible transforms; do not silently warp a whole scene.
                if np.linalg.norm(delta) > min(arr.shape[1:]) * 0.15:
                    delta = np.zeros(2, dtype=float)
                shifts[idx] = [float(delta[0]), float(delta[1])]
                aligned[idx] = shift(frame, delta, order=1, mode="nearest", prefilter=False)
        arr = aligned
    metadata = {"spatial_sigma_px": float(spatial_sigma), "registration_enabled": bool(register),
                "registration_mode": registration_mode, "registration_shifts_yx_px": shifts,
                "affine_transforms_2x3": transforms if registration_mode == "affine_ecc" and register else None}
    return arr, metadata
