"""Input readers for calibrated thermal arrays and ordered thermal image sequences."""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}
PREFERRED_NPZ_KEYS = ("temperature", "temperatures", "temps", "frames", "thermal", "array2")


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _as_video(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 4 and arr.shape[-1] in (3, 4):
        raise ValueError("RGB/pseudocolour thermal images are not accepted. Export single-channel radiometric temperatures or greyscale intensity arrays.")
    if arr.ndim != 3:
        raise ValueError(f"Expected (frames, height, width) or (height, width); got {arr.shape}")
    if min(arr.shape) < 2:
        raise ValueError(f"Invalid sequence dimensions: {arr.shape}")
    return arr.astype(np.float32, copy=False)


def npz_keys(path: str | Path) -> list[str]:
    with np.load(path, allow_pickle=False) as data:
        return list(data.files)


def load_array(path: str | Path, key: str | None = None) -> np.ndarray:
    """Load .npy, .npz, multi-page TIFF, or a 2-D CSV matrix."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _as_video(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            available = list(data.files)
            selected = key
            if selected is None:
                selected = next((k for k in PREFERRED_NPZ_KEYS if k in available), None)
            if selected is None:
                candidates = [k for k in available if np.asarray(data[k]).ndim in (2, 3)]
                if len(candidates) == 1:
                    selected = candidates[0]
                else:
                    raise ValueError(
                        "Could not select a temperature array from NPZ. "
                        f"Available keys: {available}. Pass --npz-key explicitly."
                    )
            if selected not in available:
                raise KeyError(f"NPZ key {selected!r} not found. Available keys: {available}")
            return _as_video(data[selected])
    if suffix in {".tif", ".tiff"}:
        import tifffile
        return _as_video(tifffile.imread(path))
    if suffix == ".csv":
        return _as_video(np.loadtxt(path, delimiter=","))
    raise ValueError(f"Unsupported file type: {suffix}. Use NPY, NPZ, TIFF, or CSV.")


def load_image_folder(folder: str | Path) -> np.ndarray:
    folder = Path(folder)
    files = sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES), key=lambda p: natural_key(p.name))
    if not files:
        raise ValueError(f"No supported image frames found in {folder}")
    frames = []
    shape = None
    for file in files:
        with Image.open(file) as im:
            if im.mode in {"RGB", "RGBA", "CMYK", "P"}:
                raise ValueError(f"{file.name} is a colour/pseudocolour image. Export scalar temperature data instead.")
            # Preserve integer greyscale levels, including 16-bit thermal intensities.
            frame = np.asarray(im.convert("I"), dtype=np.float32)
        if shape is None:
            shape = frame.shape
        if frame.shape != shape:
            raise ValueError(f"Frame size mismatch: {file.name} has {frame.shape}, expected {shape}")
        frames.append(frame)
    return np.stack(frames)


def load_image_zip(payload: bytes) -> np.ndarray:
    """Load image files from an uploaded ZIP without extracting untrusted paths."""
    frames: list[np.ndarray] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = sorted(
            (n for n in archive.namelist() if Path(n).suffix.lower() in IMAGE_SUFFIXES and not n.endswith("/")),
            key=natural_key,
        )
        for name in names:
            with Image.open(io.BytesIO(archive.read(name))) as im:
                if im.mode in {"RGB", "RGBA", "CMYK", "P"}:
                    raise ValueError(f"{name} is a colour/pseudocolour image. Export scalar temperature data instead.")
                frames.append(np.asarray(im.convert("I"), dtype=np.float32))
    if not frames:
        raise ValueError("ZIP contains no supported image frames")
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise ValueError("Image frames in ZIP do not all have the same dimensions")
    return np.stack(frames)


def load_sequence(path: str | Path, npz_key: str | None = None) -> np.ndarray:
    path = Path(path)
    if path.is_dir():
        return load_image_folder(path)
    if path.suffix.lower() == ".zip":
        return load_image_zip(path.read_bytes())
    return load_array(path, key=npz_key)
