
#!/usr/bin/env python3
"""
Minimal PLY colorizer: projects (x,y) onto a top‑down BMP, samples RGB, and
writes colors back into the PLY. The core steps are factored into small,
reusable functions with concise docstrings and type hints.
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from plyfile import PlyData, PlyElement

# --- Type aliases for clarity ---
Float1D = NDArray[np.float32]
Bool1D = NDArray[np.bool_]
Int1D = NDArray[np.int64]
RGBImage = NDArray[np.uint8]        # (H, W, 3)
RGBNx3 = NDArray[np.uint8]          # (N, 3)


# --------------------------- CLI --------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments.

    Returns an argparse.Namespace with:
      - input_ply: path to binary_little_endian PLY
      - input_bmp: path to top‑down RGB BMP
      - pct:      two floats (LOW HIGH) used as robust percentiles for bounds
    """
    ap = argparse.ArgumentParser(
        description="Colorize a PLY by sampling a top-down BMP."
    )
    ap.add_argument("input_ply", help="Input PLY path")
    ap.add_argument("input_bmp", help="Top-down BMP (RGB)")
    ap.add_argument(
        "--pct",
        type=float,
        nargs=2,
        default=[5.0, 95.0],
        metavar=("LOW", "HIGH"),
        help="Robust percentile window for building (x,y) bounds (default 5 95)",
    )
    return ap.parse_args()


# ------------------------ I/O utilities -------------------------------------

def read_ply_vertices(path: str) -> tuple[PlyData, np.ndarray]:
    """Load PLY and return (PlyData, vertex structured array).

    Raises SystemExit if required vertex fields are missing.
    """
    ply = PlyData.read(path)
    verts = ply["vertex"].data
    required = ("x", "y", "z", "red", "green", "blue")
    names = set(verts.dtype.names or ())
    missing = [n for n in required if n not in names]
    if missing:
        raise SystemExit(
            f"PLY vertex fields missing: {missing}; present: {sorted(names)}"
        )
    return ply, verts


def read_bmp_rgb(path: str) -> RGBImage:
    """Read an image file as RGB uint8 array of shape (H, W, 3)."""
    bmp = Image.open(path).convert("RGB")
    return np.asarray(bmp, dtype=np.uint8)


# -------------------- Geometry → Image mapping ------------------------------

def build_mask_and_bounds(
    xs: Float1D,
    ys: Float1D,
    pct: Sequence[float],
    plausible_cap: float = 1e4,
    overscan: float = 0.02,
) -> tuple[Bool1D, float, float, float, float]:
    """Create a robust mapping mask and (x,y) bounds.

    - Mask keeps finite points and discards extreme outliers via |x|,|y| < cap.
    - Bounds are computed from percentiles on the masked subset and expanded by
      a small overscan to avoid edge clipping.
    Returns: (mask, min_x, max_x, min_y, max_y)
    """
    finite = np.isfinite(xs) & np.isfinite(ys)
    plausible = (np.abs(xs) < plausible_cap) & (np.abs(ys) < plausible_cap)
    mask: Bool1D = finite & plausible
    if not mask.any():
        raise SystemExit("No plausible x/y values; cannot map to image.")

    low, high = float(pct[0]), float(pct[1])
    x_lo, x_hi = np.percentile(xs[mask], [low, high]).astype(np.float32)
    y_lo, y_hi = np.percentile(ys[mask], [low, high]).astype(np.float32)

    ov = np.float32(overscan)
    dx = (x_hi - x_lo) * ov
    dy = (y_hi - y_lo) * ov
    min_x = float(x_lo - dx)
    max_x = float(x_hi + dx)
    min_y = float(y_lo - dy)
    max_y = float(y_hi + dy)
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

    - Uses top-left image origin by default (flip_v=False ⇒ v axis is flipped).
    - Nearest-neighbor sampling: round to nearest pixel and clip to bounds.
    """
    Wf = np.float32(W - 1)
    Hf = np.float32(H - 1)

    eps = np.float32(1e-6)
    denx = np.maximum(np.float32(max_x - min_x), eps)
    deny = np.maximum(np.float32(max_y - min_y), eps)

    # Pre-allocate float arrays for clarity; fill only masked entries
    u = np.empty_like(xs, dtype=np.float32)
    v = np.empty_like(ys, dtype=np.float32)
    u[~mask] = np.nan
    v[~mask] = np.nan

    u[mask] = (xs[mask] - np.float32(min_x)) / denx * Wf
    if flip_v:
        v[mask] = (ys[mask] - np.float32(min_y)) / deny * Hf
    else:
        v[mask] = (np.float32(1.0) - (ys[mask] - np.float32(min_y)) / deny) * Hf

    # Nearest-neighbor indices (int64) with clipping
    uu: Int1D = np.zeros_like(xs, dtype=np.int64)
    vv: Int1D = np.zeros_like(ys, dtype=np.int64)
    uu[mask] = np.clip(np.rint(u[mask]), 0, W - 1).astype(np.int64)
    vv[mask] = np.clip(np.rint(v[mask]), 0, H - 1).astype(np.int64)
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
    rgb: RGBNx3 = np.empty((N, 3), dtype=np.uint8)
    rgb[:] = np.array(default_rgb, dtype=np.uint8)
    rgb[mask] = bmp[vv[mask], uu[mask]]  # vectorized gather
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

    # Sparse reconstruction: paint sampled colors onto a black canvas
    recon = np.zeros_like(bmp)
    recon[vv[mask], uu[mask]] = rgb[mask]
    Image.fromarray(recon, "RGB").save(recon_path)

    # Coverage mask: white where any point lands
    cover = np.zeros((H, W, 3), dtype=np.uint8)
    cover[vv[mask], uu[mask]] = 255
    Image.fromarray(cover, "RGB").save(cover_path)


def write_output_ply(verts: np.ndarray, rgb: RGBNx3, out_path: str = "output.ply") -> None:
    """Write colored vertices into a new binary PLY file."""
    verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(out_path)


# ------------------------------ Main ----------------------------------------

def main() -> int:
    args = parse_args()

    print("[1/6] Reading PLY:", args.input_ply)
    _, verts = read_ply_vertices(args.input_ply)

    print("[2/6] Reading BMP:", args.input_bmp)
    bmp_arr = read_bmp_rgb(args.input_bmp)
    H, W = bmp_arr.shape[:2]

    print("[3/6] Mapping vertices to BMP colors…")
    xs: Float1D = verts["x"].astype(np.float32, copy=False)
    ys: Float1D = verts["y"].astype(np.float32, copy=False)

    mask, min_x, max_x, min_y, max_y = build_mask_and_bounds(xs, ys, args.pct)
    uu, vv = map_xy_to_pixels(xs, ys, mask, H, W, min_x, max_x, min_y, max_y)

    print("[4/6] Sampling colors…")
    rgb = sample_colors(bmp_arr, uu, vv, mask, N=verts.shape[0])

    print("[5/6] Writing output PLY: output.ply")
    write_output_ply(verts, rgb, out_path="output.ply")

    try:
        print("[6/6] Writing preview images…")
        write_previews(bmp_arr, uu, vv, mask, rgb)
        print("   Wrote preview_reconstruct.bmp and preview_coverage.bmp")
    except Exception as e:
        print(f"[warn] Failed to write previews: {e}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
