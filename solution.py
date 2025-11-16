#!/usr/bin/env python3
"""
Minimal PLY colorizer with pragmatic safety + optional LUT correction.

Pipeline:
  1) Read binary_little_endian PLY and top-down BMP (RGB).
  2) Build robust XY bounds (percentiles with overscan), map (x,y) → (u,v).
  3) Nearest-neighbor sample BMP at (u,v); write RGB to vertices.
  4) (Optional) Apply LUT color normalization as a function of (X,Z).
  5) Save colored PLY + two quick preview BMPs.

LUT format (text), example lines:
  -576 576 399 1551
  X: -85 Z: 400 C: 693 684 547 835 0 0 537
First line: xmin xmax zmin zmax (informational).
Subsequent lines: integer X, integer Z, then 7 channel factors.
Zeros are treated as missing (ignored); nearest neighbor sampling is used.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import NoReturn, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from plyfile import PlyData, PlyElement
import re

# --- Config globals ---
# Robust percentile window for bounds (no CLI override)
PCT_LOW: float = 0.0
PCT_HIGH: float = 100.0
# Small margin around the percentile box
OVERSCAN: float = 0.02
# Hard plausibility cap for XY (absolute units of scanner space)
PLAUSIBLE_CAP: float = 1e4

# LUT behavior
LUT_CHANNELS: Tuple[int, int, int] = (0, 1, 2)   # which of the 7 C-values map to R,G,B
LUT_MODE: str = "divide"                         # "divide" for normalization factors; "multiply" for gains
LUT_EPS: float = 1e-9                            # protect against divide-by-zero
LUT_METHOD: str = "nearest"                      # keep minimal & robust: "nearest"


# --- Type aliases ---
Float1D = NDArray[np.float32]
Bool1D = NDArray[np.bool_]
Int1D = NDArray[np.int64]
RGBImage = NDArray[np.uint8]        # (H, W, 3)
RGBNx3  = NDArray[np.uint8]         # (N, 3)


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
      - lut:       optional LUT file path
    """
    ap = argparse.ArgumentParser(description="Colorize a PLY by sampling a top-down BMP.")
    ap.add_argument("input_ply", help="Input PLY path")
    ap.add_argument("input_bmp", help="Top-down BMP (RGB)")
    ap.add_argument("--lut", help="Optional LUT file for RGB normalization (text format)", default=None)
    return ap.parse_args()


# ------------------------ I/O utilities -------------------------------------
def read_ply_vertices(path: str) -> tuple[PlyData, np.ndarray]:
    """Load PLY and return (PlyData, vertex structured array)."""
    if not os.path.exists(path):
        die(f"PLY not found: {path}")
    try:
        ply = PlyData.read(path)
    except Exception as e:
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
    """Create a robust mapping mask and (x,y) bounds using global config."""
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
    """Map (x,y) to integer pixel indices (uu,vv) using linear normalization."""
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
    """Gather RGB from image at (vv, uu) for masked vertices; others get default."""
    if uu.shape != vv.shape or uu.shape != mask.shape:
        die("uu/vv/mask shape mismatch")

    rgb: RGBNx3 = np.empty((N, 3), dtype=np.uint8)
    rgb[:] = np.array(default_rgb, dtype=np.uint8)
    try:
        rgb[mask] = bmp[vv[mask], uu[mask]]  # vectorized gather
    except Exception as e:
        die(f"Color sampling failed (indexing image): {e}")
    return rgb


# ---------------------------- LUT support -----------------------------------
class LutGrid:
    """Sparse 2D LUT over integer X,Z with per-channel factors (3 chosen channels)."""
    def __init__(self, xs: np.ndarray, zs: np.ndarray, factors: np.ndarray) -> None:
        # xs: [Nx] int32 sorted; zs: [Nz] int32 sorted; factors: [Nz, Nx, 3] float64 (NaN for missing)
        self.xs = xs
        self.zs = zs
        self.factors = factors

    @staticmethod
    def _idx_nearest(arr: np.ndarray, vals: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(arr, vals, side="left")
        pos = np.clip(pos, 0, len(arr) - 1)
        # choose left neighbor if closer (or equal distance)
        left = (pos > 0) & ((pos == len(arr)) | (np.abs(vals - arr[pos - 1]) <= np.abs(arr[pos] - vals)))
        pos[left] -= 1
        return pos

    def sample_nearest(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Nearest-neighbor sampling; returns [N,3] float64 factors (NaN where missing)."""
        xv = np.asarray(x, dtype=np.float64).reshape(-1)
        zv = np.asarray(z, dtype=np.float64).reshape(-1)
        # clamp to grid range before nearest
        xv = np.clip(xv, self.xs[0], self.xs[-1])
        zv = np.clip(zv, self.zs[0], self.zs[-1])
        xi = self._idx_nearest(self.xs, xv)
        zi = self._idx_nearest(self.zs, zv)
        return self.factors[zi, xi]  # [N,3]


def load_lut(path: str, channels: Tuple[int, int, int] = LUT_CHANNELS) -> LutGrid:
    """Parse LUT text file; return a LutGrid with chosen 3 channels."""
    if not os.path.exists(path):
        die(f"LUT not found: {path}")

    rx_header = re.compile(r"^\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$")
    rx_entry  = re.compile(r"^\s*X:\s*(-?\d+)\s+Z:\s*(-?\d+)\s+C:\s*([0-9\s\-]+)\s*$")

    rows: dict[Tuple[int, int], np.ndarray] = {}
    header: Optional[Tuple[int, int, int, int]] = None

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = rx_entry.match(line)
                if m:
                    x = int(m.group(1))
                    z = int(m.group(2))
                    nums = [int(t) for t in m.group(3).split()]
                    # ensure length ≥ max(channels)+1
                    need = max(channels) + 1
                    if len(nums) < need:
                        nums += [0] * (need - len(nums))
                    vals = np.array([nums[i] for i in channels], dtype=np.float64)
                    # treat all-zeros as missing
                    if np.all(vals == 0):
                        vals[:] = np.nan
                    rows[(x, z)] = vals
                    continue
                m = rx_header.match(line)
                if m:
                    header = tuple(int(g) for g in m.groups())
                    continue
                # ignore anything else
    except Exception as e:
        die(f"Failed to read LUT '{path}': {e}")

    if not rows:
        die("No LUT entries parsed.")

    xs = np.array(sorted({x for (x, _) in rows.keys()}), dtype=np.int32)
    zs = np.array(sorted({z for (_, z) in rows.keys()}), dtype=np.int32)

    Nx, Nz = len(xs), len(zs)
    fac = np.full((Nz, Nx, 3), np.nan, dtype=np.float64)
    xi_index = {x: i for i, x in enumerate(xs)}
    zi_index = {z: i for i, z in enumerate(zs)}
    for (x, z), v in rows.items():
        fac[zi_index[z], xi_index[x]] = v

    # Simple stats print (optional but helpful)
    cmax = np.nanmax(fac, axis=(0, 1))
    xr = (int(xs[0]), int(xs[-1]))
    zr = (int(zs[0]), int(zs[-1]))
    cmax_py = [int(v) if np.isfinite(v) else 0 for v in cmax]
    print(f"   LUT ranges  X[{xr[0]},{xr[1]}]  Z[{zr[0]},{zr[1]}]  c_max={cmax_py}")

    return LutGrid(xs=xs, zs=zs, factors=fac)


def apply_lut_rgb(
    lut: LutGrid,
    x: np.ndarray,   # world/scanner X per vertex
    z: np.ndarray,   # world/scanner Z per vertex
    rgb: RGBNx3,
    mode: str = LUT_MODE,
    eps: float = LUT_EPS,
    method: str = LUT_METHOD,
) -> RGBNx3:
    """Apply per-vertex LUT normalization to RGB. Returns uint8 RGB."""
    if method != "nearest":
        # minimal implementation: keep nearest only for robustness/speed
        method = "nearest"

    fac = lut.sample_nearest(x, z)  # [N,3] float64 (NaN where missing)
    out = rgb.astype(np.float64, copy=False)

    # validity per channel
    valid = (~np.isnan(fac)) & (np.abs(fac) > eps)
    safe = np.where(valid, fac, 1.0)

    if mode == "divide":
        out = out / safe
    else:
        out = out * safe

    return np.clip(out, 0, 255).astype(np.uint8)


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

        if args.lut:
            print("[5/6] Loading LUT and applying normalization…")
            lut = load_lut(args.lut, channels=LUT_CHANNELS)
            # Use world/scanner coordinates X,Z from the PLY (finite subset only to avoid warnings)
            x_all = verts["x"]
            z_all = verts["z"]
            mask_lut = mask & np.isfinite(x_all) & np.isfinite(z_all)
            if mask_lut.any():
                xv = x_all[mask_lut].astype(np.float64, copy=False)
                zv = z_all[mask_lut].astype(np.float64, copy=False)
                rgb_part = apply_lut_rgb(
                    lut, xv, zv, rgb[mask_lut], mode=LUT_MODE, eps=LUT_EPS, method=LUT_METHOD
                )
                rgb = rgb.copy()
                rgb[mask_lut] = rgb_part
            print("   LUT correction applied.")

            print("[6/6] Writing output PLY: output.ply")
        else:
            print("[5/6] Writing output PLY: output.ply")

        write_output_ply(verts, rgb, out_path="output.ply")

        print("   Writing preview images…")
        write_previews(bmp_arr, uu, vv, mask, rgb)

        print("Done.")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())