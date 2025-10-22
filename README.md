# PLY Colorizer (minimal)

Colorizes a binary-little-endian PLY point cloud using a top-down BMP.  
It projects each vertex’s (x, y) to image pixels, samples RGB, writes back to the PLY, and saves quick preview images.

## Run

Using **uv**:

First off, if you don't have ```pyproject.toml```, run:
```bash
uv init
```

After that, add all tools needed:
```bash
uv add numpy pillow plyfile
```

After everything is installed, you can run the ```solution.py```:
```bash
uv run solution.py <input.ply> <image.bmp> [--pct LOW HIGH]
```
Example:
```bash
uv run solution.py sample_gray.ply sample.ply.bmp --pct 5 95
```

## Output
1. output.ply — recolored point cloud
2. preview_reconstruct.bmp — sparse image reconstructed from points
3. preview_coverage.bmp — white mask where points landed
