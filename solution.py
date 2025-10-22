#!/usr/bin/env python3
import argparse, re, sys
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import open3d as o3d


def parse_args():
    ap = argparse.ArgumentParser(
        description="Colorize a PLY by sampling a top-down BMP, using ms_lut for u (column) and Y→v (row)."
    )
    ap.add_argument("input_ply", help="Input PLY path")
    ap.add_argument("input_bmp", help="Top-down BMP (RGB)")
    ap.add_argument("--pct", type=float, nargs=2, default=[5.0, 95.0],
                    metavar=("LOW","HIGH"),
                    help="Robust percentile window for Y→v scaling (default 5 95)")
    return ap.parse_args()



def main():
    args = parse_args()

    print("[1/6] Reading PLY:", args.input_ply)
    ply = PlyData.read(args.input_ply)
    verts = ply["vertex"].data
    names = verts.dtype.names
    need = ("x","y","z","red","green","blue")
    for n in need:
        if n not in names:
            raise SystemExit(f"PLY vertex field '{n}' missing; fields present: {names}")
    
    

    print("[2/6] Reading BMP:", args.input_bmp)
    bmp = Image.open(args.input_bmp).convert("RGB")
    bmp_arr = np.asarray(bmp, dtype=np.uint8)
    H, W = bmp_arr.shape[:2]
    print(f"   BMP size: {W}x{H}")


    xs = verts["x"].astype(np.float32, copy=False)
    ys = verts["y"].astype(np.float32, copy=False)

    finite = np.isfinite(xs) & np.isfinite(ys)
    plausible = (np.abs(xs) < 1e4) & (np.abs(ys) < 1e4)
    mask = finite & plausible
    if not mask.any():
        raise SystemExit("No plausible x/y values; cannot map to image.")

    low, high = args.pct
    x_lo, x_hi = np.percentile(xs[mask], [low, high]).astype(np.float32)
    y_lo, y_hi = np.percentile(ys[mask], [low, high]).astype(np.float32)
    
    overscan = np.float32(0.02)
    dx = (x_hi - x_lo) * overscan; dy = (y_hi - y_lo) * overscan
    min_x = x_lo - dx; max_x = x_hi + dx
    min_y = y_lo - dy; max_y = y_hi + dy


    Wf = np.float32(W - 1); Hf = np.float32(H - 1)
    
    # Prevent diving by zero if all x or all y are identical
    eps = np.float32(1e-6)
    denx = np.maximum(max_x - min_x, eps)
    deny = np.maximum(max_y - min_y, eps)



    u = np.empty_like(xs, dtype=np.float32)
    v = np.empty_like(ys, dtype=np.float32)
    u[~mask] = np.nan
    v[~mask] = np.nan

    u[mask] = (xs[mask] - min_x) / denx * Wf
    v[mask] = (np.float32(1.0) - (ys[mask] - min_y) / deny) * Hf


    uu = np.zeros_like(xs, dtype=np.int64)
    vv = np.zeros_like(ys, dtype=np.int64)
    uu[mask] = np.clip(np.rint(u[mask]), 0, W - 1).astype(np.int64)
    vv[mask] = np.clip(np.rint(v[mask]), 0, H - 1).astype(np.int64)

    rgb = np.zeros((verts.shape[0], 3), dtype=np.uint8)
    rgb[mask] = bmp_arr[vv[mask], uu[mask]]
            

    verts["red"]   = rgb[:, 0]
    verts["green"] = rgb[:, 1]
    verts["blue"]  = rgb[:, 2]    

    try:
        # Reconstructed image from projected points
        recon = np.zeros_like(bmp_arr)
        recon[vv[mask], uu[mask]] = rgb[mask]  # paint sampled colors where we have points
        Image.fromarray(recon, 'RGB').save('preview_reconstruct.bmp')

        # Coverage mask (white where a point landed)
        cover = np.zeros((H, W, 3), dtype=np.uint8)
        cover[vv[mask], uu[mask]] = 255
        Image.fromarray(cover, 'RGB').save('preview_coverage.bmp')

        print("   Wrote preview_reconstruct.bmp and preview_coverage.bmp")
    except Exception as e:
        print(f"[warn] Failed to write previews: {e}")

    PlyData([PlyElement.describe(verts, "vertex")], text=False).write("output.ply")
    print("Done.")
    return 0
    

if __name__ == "__main__":
    sys.exit(main())
    
