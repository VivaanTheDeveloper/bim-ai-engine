"""
preprocessor.py
---------------
Converts raw IFC + DXF file pairs into
numpy tensor arrays ready for AI training.

Reads from:  dataset/raw_ifc/     and  dataset/dxf_targets/
Writes to:   dataset/processed_x/ and  dataset/processed_y/
"""

import numpy as np
import ezdxf
from pathlib import Path
from geometry_engine import extract_geometry

CANVAS_SIZE = 256   # Neural network input resolution (pixels)

def ifc_to_tensor(ifc_path: str) -> np.ndarray:
    """Convert one IFC file into a 256x256 wall geometry matrix."""
    package = extract_geometry(ifc_path)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)

    for wall in package.walls:
        # Normalize wall coordinates to 256x256 pixel space
        x1 = int((wall.x1 / 50000.0) * (CANVAS_SIZE - 1))
        y1 = int((wall.y1 / 50000.0) * (CANVAS_SIZE - 1))
        x2 = int((wall.x2 / 50000.0) * (CANVAS_SIZE - 1))
        y2 = int((wall.y2 / 50000.0) * (CANVAS_SIZE - 1))

        # Draw the wall as a line on the canvas using Bresenham's algorithm
        _draw_line_on_canvas(canvas, x1, y1, x2, y2)

    return canvas.reshape(1, CANVAS_SIZE, CANVAS_SIZE)  # Shape: [1, 256, 256]


def dxf_to_tensor(dxf_path: str) -> np.ndarray:
    """Convert one DXF drawing into a 256x256 annotation heatmap using absolute scale matching."""
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)

    try:
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()

        dim_points = []
        for entity in msp:
            if entity.dxftype() in ("DIMENSION", "TEXT", "MTEXT"):
                if hasattr(entity.dxf, 'defpoint'):
                    dim_points.append(entity.dxf.defpoint[:2])

        for px, py in dim_points:
            # CRITICAL FIX: Match the identical 50000.0mm global scaling rule used by your IFC script
            cx = int((px / 50000.0) * (CANVAS_SIZE - 1))
            cy = int((py / 50000.0) * (CANVAS_SIZE - 1))
            
            cx = max(0, min(CANVAS_SIZE - 1, cx))
            cy = max(0, min(CANVAS_SIZE - 1, cy))
            canvas[cy, cx] = 1.0

    except Exception as e:
        print(f"  Warning: Could not parse DXF {dxf_path}: {e}")

    return canvas.reshape(1, CANVAS_SIZE, CANVAS_SIZE)


def _draw_line_on_canvas(canvas, x1, y1, x2, y2):
    """Bresenham line algorithm — draws a pixel-perfect line on a numpy canvas."""
    x1 = max(0, min(CANVAS_SIZE - 1, x1))
    y1 = max(0, min(CANVAS_SIZE - 1, y1))
    x2 = max(0, min(CANVAS_SIZE - 1, x2))
    y2 = max(0, min(CANVAS_SIZE - 1, y2))

    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        canvas[y1, x1] = 1.0
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy


def run_preprocessing_pipeline():
    """
    Master pipeline — processes all IFC/DXF pairs and
    saves tensor arrays to processed_x/ and processed_y/
    """
    ifc_dir = Path("dataset/raw_ifc")
    dxf_dir = Path("dataset/dxf_targets")
    out_x   = Path("dataset/processed_x")
    out_y   = Path("dataset/processed_y")

    ifc_files = sorted(ifc_dir.glob("*.ifc"))
    dxf_files = sorted(dxf_dir.glob("*.dxf"))

    pairs = min(len(ifc_files), len(dxf_files))

    if pairs == 0:
        print("No IFC/DXF pairs found.")
        print(f"  Put .ifc files in: {ifc_dir}/")
        print(f"  Put .dxf files in: {dxf_dir}/")
        return

    print(f"Found {pairs} training pairs. Starting preprocessing...")

    for i in range(pairs):
        print(f"  [{i+1}/{pairs}] {ifc_files[i].name} + {dxf_files[i].name}")

        x_tensor = ifc_to_tensor(str(ifc_files[i]))
        y_tensor  = dxf_to_tensor(str(dxf_files[i]))

        np.save(str(out_x / f"sample_{i:04d}.npy"), x_tensor)
        np.save(str(out_y / f"sample_{i:04d}.npy"), y_tensor)

    print(f"\nPreprocessing complete. {pairs} tensor pairs saved.")
    print(f"  Inputs  → {out_x}/")
    print(f"  Targets → {out_y}/")


if __name__ == "__main__":
    run_preprocessing_pipeline()