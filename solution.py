#!/usr/bin/env python3
"""
Minimal PLY colorizer with pragmatic safety:
- Projects (x,y) onto a top-down BMP, samples RGB, writes colors back.
- Adds simple, targeted error handling around file I/O and fragile steps.
- Keeps code small and readable; no heavy frameworks.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import NoReturn, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from plyfile import PlyData, PlyElement

# --- Config globals ---
# Robust percentile window for bounds (no CLI override)
PCT_LOW: float = 0.0
PCT_HIGH: float = 100.0
# Small margin around the percentile box
OVERSCAN: float = 0.02
# Hard plausibility cap for XY (absolute units of scanner space)
PLAUSIBLE_CAP: float = 1e4

# --- Type aliases ---
Float1D = NDArray[np.float32]
Bool1D = NDArray[np.bool_]
Int1D = NDArray[np.int64]
RGBImage = NDArray[np.uint8]        # (H, W, 3)
RGBNx3 = NDArray[np.uint8]          # (N, 3)


# --------------------------- small helpers ----------------------------------
def die(msg: str, code: int = 2) -> NoReturn:
    """Print an error message and exit with a non-zero code."""
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


# --------------------------- CLI --------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns an argparse.Namespace with:
      - input_ply: path to binary_little_endian PLY
      - input_bmp: path to top-down RGB BMP
    """
    ap = argparse.ArgumentParser(description="Colorize a PLY by sampling a top-down BMP.")
    ap.add_argument("input_ply", help="Input PLY path")
    ap.add_argument("input_bmp", help="Top-down BMP (RGB)")
    return ap.parse_args()


# ------------------------ I/O utilities -------------------------------------
def read_ply_vertices(path: str) -> tuple[PlyData, np.ndarray]:
    """Load PLY and return (PlyData, vertex structured array).

    Exits early with a clear message on common I/O/format errors.
    """
    if not os.path.exists(path):
        die(f"PLY not found: {path}")
    try:
        ply = PlyData.read(path)
    except Exception as e:  # file corrupt / not a PLY / permissions, etc.
        die(f"Failed to read PLY '{path}': {e}")
    try:
        verts = ply["vertex"].data
    except Exception:
        die("PLY missing 'vertex' element")

    names = set(verts.dtype.names or ())
    required = {"x", "y", "z", "red", "green", "blue"}
    missing = sorted(required - names)
    if missing:
        die(f"PLY vertex fields missing: {missing}; present: {sorted(names)}")
    return ply, verts


def read_bmp_rgb(path: str) -> RGBImage:
    """Read an image file as RGB uint8 array of shape (H, W, 3)."""
    if not os.path.exists(path):
        die(f"BMP not found: {path}")
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)
    except Exception as e:
        die(f"Failed to read BMP '{path}': {e}")
    if arr.ndim != 3 or arr.shape[2] != 3:
        die(f"Unexpected BMP shape {arr.shape}; expected (H,W,3) RGB")
    H, W = arr.shape[:2]
    if H <= 0 or W <= 0:
        die("BMP has non-positive dimensions")
    return arr


# -------------------- Geometry → Image mapping ------------------------------
def build_mask_and_bounds(
    xs: Float1D,
    ys: Float1D,
) -> tuple[Bool1D, float, float, float, float]:
    """Create a robust mapping mask and (x,y) bounds using global config.

    - Mask keeps finite points and discards extreme outliers via |x|,|y| < PLAUSIBLE_CAP.
    - Bounds come from percentiles [PCT_LOW, PCT_HIGH] on the masked subset, expanded by
      a small OVERSCAN to avoid edge clipping.
    Returns: (mask, min_x, max_x, min_y, max_y)
    """
    if xs.size == 0 or ys.size == 0:
        die("Empty vertex arrays")

    finite = np.isfinite(xs) & np.isfinite(ys)
    plausible = (np.abs(xs) < PLAUSIBLE_CAP) & (np.abs(ys) < PLAUSIBLE_CAP)
    mask: Bool1D = finite & plausible
    if not mask.any():
        die("No plausible x/y values; cannot map to image.")

    try:
        low, high = float(PCT_LOW), float(PCT_HIGH)
        if not (0.0 <= low < high <= 100.0):
            die(f"Invalid percentile window: {[PCT_LOW, PCT_HIGH]}")
        x_lo, x_hi = np.percentile(xs[mask], [low, high]).astype(np.float32)
        y_lo, y_hi = np.percentile(ys[mask], [low, high]).astype(np.float32)
    except Exception as e:
        die(f"Percentile computation failed: {e}")

    ov = np.float32(OVERSCAN)
    dx = (x_hi - x_lo) * ov
    dy = (y_hi - y_lo) * ov
    min_x = float(x_lo - dx)
    max_x = float(x_hi + dx)
    min_y = float(y_lo - dy)
    max_y = float(y_hi + dy)

    if not np.isfinite([min_x, max_x, min_y, max_y]).all():
        die("Non-finite mapping bounds computed")
    if (max_x - min_x) <= 0 or (max_y - min_y) <= 0:
        die("Degenerate mapping bounds (zero span)")

    return mask, min_x, max_x, min_y, max_y


def map_xy_to_pixels(
    xs: Float1D,
    ys: Float1D,
    mask: Bool1D,
    H: int,
    W: int,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    *,
    flip_v: bool = False,
) -> tuple[Int1D, Int1D]:
    """Map (x,y) to integer pixel indices (uu,vv) using linear normalization.

    - Top-left image origin by default (flip_v=False ⇒ v axis flipped).
    - Nearest-neighbor sampling: round to nearest pixel and clip to bounds.
    """
    if H <= 0 or W <= 0:
        die("Image has invalid dimensions")

    Wf = np.float32(W - 1)
    Hf = np.float32(H - 1)

    eps = np.float32(1e-6)
    denx = np.maximum(np.float32(max_x - min_x), eps)
    deny = np.maximum(np.float32(max_y - min_y), eps)

    u = np.empty_like(xs, dtype=np.float32)
    v = np.empty_like(ys, dtype=np.float32)
    u[~mask] = np.nan
    v[~mask] = np.nan

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        u[mask] = (xs[mask] - np.float32(min_x)) / denx * Wf
        if flip_v:
            v[mask] = (ys[mask] - np.float32(min_y)) / deny * Hf
        else:
            v[mask] = (np.float32(1.0) - (ys[mask] - np.float32(min_y)) / deny) * Hf

    uu: Int1D = np.zeros_like(xs, dtype=np.int64)
    vv: Int1D = np.zeros_like(ys, dtype=np.int64)
    try:
        uu[mask] = np.clip(np.rint(u[mask]), 0, W - 1).astype(np.int64)
        vv[mask] = np.clip(np.rint(v[mask]), 0, H - 1).astype(np.int64)
    except Exception as e:
        die(f"Index rounding/clipping failed: {e}")
    return uu, vv


def sample_colors(
    bmp: RGBImage,
    uu: Int1D,
    vv: Int1D,
    mask: Bool1D,
    N: int,
    *,
    default_rgb: Tuple[int, int, int] = (0, 0, 0),
) -> RGBNx3:
    """Gather RGB from image at (vv, uu) for masked vertices; others get default.
    Returns an (N,3) uint8 array.
    """
    if uu.shape != vv.shape or uu.shape != mask.shape:
        die("uu/vv/mask shape mismatch")

    rgb: RGBNx3 = np.empty((N, 3), dtype=np.uint8)
    rgb[:] = np.array(default_rgb, dtype=np.uint8)
    try:
        rgb[mask] = bmp[vv[mask], uu[mask]]  # vectorized gather
    except Exception as e:
        die(f"Color sampling failed (indexing image): {e}")
    return rgb


# ---------------------------- Outputs ---------------------------------------
def write_previews(
    bmp: RGBImage,
    uu: Int1D,
    vv: Int1D,
    mask: Bool1D,
    rgb: RGBNx3,
    *,
    recon_path: str = "preview_reconstruct.bmp",
    cover_path: str = "preview_coverage.bmp",
) -> None:
    """Save two quicklook BMPs: a sparse reconstruction and a coverage mask."""
    H, W = bmp.shape[:2]
    try:
        recon = np.zeros_like(bmp)
        recon[vv[mask], uu[mask]] = rgb[mask]
        Image.fromarray(recon, "RGB").save(recon_path)

        cover = np.zeros((H, W, 3), dtype=np.uint8)
        cover[vv[mask], uu[mask]] = 255
        Image.fromarray(cover, "RGB").save(cover_path)
    except Exception as e:
        print(f"[warn] Failed to write previews: {e}")


def write_output_ply(verts: np.ndarray, rgb: RGBNx3, out_path: str = "output.ply") -> None:
    """Write colored vertices into a new binary PLY file."""
    try:
        verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        PlyData([PlyElement.describe(verts, "vertex")], text=False).write(out_path)
    except Exception as e:
        die(f"Failed to write output PLY '{out_path}': {e}")


# ------------------------------ Main ----------------------------------------
def main() -> int:
    try:
        args = parse_args()

        print("[1/6] Reading PLY:", args.input_ply)
        _, verts = read_ply_vertices(args.input_ply)

        print("[2/6] Reading BMP:", args.input_bmp)
        bmp_arr = read_bmp_rgb(args.input_bmp)
        H, W = bmp_arr.shape[:2]

        print("[3/6] Mapping vertices to BMP colors…")
        xs: Float1D = verts["x"].astype(np.float32, copy=False)
        ys: Float1D = verts["y"].astype(np.float32, copy=False)

        mask, min_x, max_x, min_y, max_y = build_mask_and_bounds(xs, ys)
        uu, vv = map_xy_to_pixels(xs, ys, mask, H, W, min_x, max_x, min_y, max_y)

        print("[4/6] Sampling colors…")
        rgb = sample_colors(bmp_arr, uu, vv, mask, N=verts.shape[0])

        print("[5/6] Writing output PLY: output.ply")
        write_output_ply(verts, rgb, out_path="output.ply")

        print("[6/6] Writing preview images…")
        write_previews(bmp_arr, uu, vv, mask, rgb)

        print("Done.")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())