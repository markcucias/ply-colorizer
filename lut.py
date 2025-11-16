# lut.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, Optional
import numpy as np
import re
import math

@dataclass
class LutGrid:
    xs: np.ndarray                 # shape: [Nx], sorted unique Xs
    zs: np.ndarray                 # shape: [Nz], sorted unique Zs
    factors: np.ndarray            # shape: [Nz, Nx, C] where C=len(use_channels)
    channel_indices: Tuple[int, int, int]

    def _idx_nearest(self, arr: np.ndarray, vals: np.ndarray) -> np.ndarray:
        # nearest-neighbor indices for vals in sorted arr
        pos = np.searchsorted(arr, vals, side="left")
        pos = np.clip(pos, 0, len(arr)-1)
        left_is_better = (pos > 0) & (
            (pos == len(arr)) | (np.abs(vals - arr[pos-1]) <= np.abs(arr[pos] - vals))
        )
        pos[left_is_better] -= 1
        return pos

    def _bilinear(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        # bilinear interpolation on integer grid; falls back to nearest if any corner is NaN
        x = x.astype(np.float64)
        z = z.astype(np.float64)

        # clamp to grid bounds
        x = np.clip(x, self.xs[0], self.xs[-1])
        z = np.clip(z, self.zs[0], self.zs[-1])

        # get surrounding grid indices
        xi1 = np.searchsorted(self.xs, x, side="left")
        xi1 = np.clip(xi1, 1, len(self.xs)-1)
        xi0 = xi1 - 1
        zi1 = np.searchsorted(self.zs, z, side="left")
        zi1 = np.clip(zi1, 1, len(self.zs)-1)
        zi0 = zi1 - 1

        x0 = self.xs[xi0]; x1 = self.xs[xi1]
        z0 = self.zs[zi0]; z1 = self.zs[zi1]

        # avoid division by zero if grid step is 0 (degenerate)
        tx = np.divide(x - x0, (x1 - x0), out=np.zeros_like(x), where=(x1 != x0))
        tz = np.divide(z - z0, (z1 - z0), out=np.zeros_like(z), where=(z1 != z0))

        f00 = self.factors[zi0, xi0]  # shape [..., 3]
        f10 = self.factors[zi0, xi1]
        f01 = self.factors[zi1, xi0]
        f11 = self.factors[zi1, xi1]

        # any NaN in the four corners -> fall back to nearest
        any_nan = (
            np.isnan(f00).any(axis=1) |
            np.isnan(f10).any(axis=1) |
            np.isnan(f01).any(axis=1) |
            np.isnan(f11).any(axis=1)
        )

        # bilinear weights
        tx = tx[:, None]
        tz = tz[:, None]
        w00 = (1 - tx) * (1 - tz)
        w10 = tx * (1 - tz)
        w01 = (1 - tx) * tz
        w11 = tx * tz
        interp = w00 * f00 + w10 * f10 + w01 * f01 + w11 * f11

        if np.any(any_nan):
            # nearest neighbor for the ones that had NaNs in their stencil
            xi_nn = self._idx_nearest(self.xs, x[any_nan])
            zi_nn = self._idx_nearest(self.zs, z[any_nan])
            interp[any_nan] = self.factors[zi_nn, xi_nn]

        return interp  # shape [N, 3]

    def sample(self, x: np.ndarray, z: np.ndarray, method: str = "bilinear") -> np.ndarray:
        """
        x,z: arrays of the same length (ints or floats)
        method: 'bilinear' or 'nearest'
        returns: factors [N,3]
        """
        x = np.asarray(x).reshape(-1)
        z = np.asarray(z).reshape(-1)
        if method == "nearest":
            xi = self._idx_nearest(self.xs, x.astype(np.float64))
            zi = self._idx_nearest(self.zs, z.astype(np.float64))
            return self.factors[zi, xi]
        return self._bilinear(x.astype(np.float64), z.astype(np.float64))

    def apply_to_rgb(
        self,
        x: np.ndarray,
        z: np.ndarray,
        rgb: np.ndarray,
        method: str = "bilinear",
        mode: str = "divide",                 # 'divide' for normalization factors, 'multiply' for gains
        eps: float = 1e-9
    ) -> np.ndarray:
        """
        Vectorized correction.
        rgb: shape [N,3] uint8 or float
        returns corrected rgb as float64 in [0, inf)
        """
        rgb = np.asarray(rgb, dtype=np.float64)
        factors = self.sample(x, z, method=method)  # [N,3]

        # where factors are NaN or 0 -> keep original
        mask_valid = (~np.isnan(factors)) & (np.abs(factors) > eps)
        out = rgb.copy()

        if mode == "divide":
            safe = np.where(mask_valid, factors, 1.0)
            out = rgb / safe
        else:
            safe = np.where(mask_valid, factors, 1.0)
            out = rgb * safe

        # clip back to 0..255 if you want uint8 later
        return np.clip(out, 0, 255)

def load_lut(
    path: str,
    use_channels: Tuple[int, int, int] = (0, 1, 2),
    normalize_to: Optional[float] = None,   # e.g., 1.0 or 255.0 to rescale factors; None leaves as-is
) -> LutGrid:
    """
    Reads a LUT text file with lines like:
      -576 576 399 1551
      X: -85 Z: 400 C: 693 684 547 835 0 0 537
    Returns an interpolatable grid. Zeros are treated as missing.
    """
    header = None
    rows: Dict[Tuple[int, int], np.ndarray] = {}

    rx_header = re.compile(r"^\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$")
    rx_entry  = re.compile(r"^\s*X:\s*(-?\d+)\s+Z:\s*(-?\d+)\s+C:\s*([0-9\s\-]+)\s*$")

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
                if len(nums) < 7:
                    # pad to 7 for robustness
                    nums += [0] * (7 - len(nums))
                # pick the three channels we’ll use
                vals = np.array([nums[i] for i in use_channels], dtype=np.float64)
                # treat all-zeros as missing
                if np.all(vals == 0):
                    vals[:] = np.nan
                rows[(x, z)] = vals
                continue
            m = rx_header.match(line)
            if m:
                header = tuple(int(g) for g in m.groups())
                continue
            # ignore any other lines/comments

    if not rows:
        raise ValueError("No LUT entries parsed.")

    xs = np.array(sorted({x for (x, _) in rows.keys()}), dtype=np.int32)
    zs = np.array(sorted({z for (_, z) in rows.keys()}), dtype=np.int32)

    Nx, Nz = len(xs), len(zs)
    fac = np.full((Nz, Nx, 3), np.nan, dtype=np.float64)

    xi_index = {x: i for i, x in enumerate(xs)}
    zi_index = {z: i for i, z in enumerate(zs)}
    for (x, z), v in rows.items():
        fac[zi_index[z], xi_index[x]] = v

    if normalize_to is not None:
        # rescale so that the 95th percentile maps to normalize_to (robust against outliers)
        finite = np.isfinite(fac)
        if finite.any():
            p95 = np.nanpercentile(fac, 95)
            if p95 > 0:
                fac = fac * (normalize_to / p95)

    # Optional: if header gives min/max that extend beyond parsed keys, we ignore it—sparse is fine.
    return LutGrid(xs=xs, zs=zs, factors=fac, channel_indices=use_channels)