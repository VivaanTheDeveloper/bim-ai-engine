"""
inference_engine.py
-------------------
Loads model. Auto-pulls from cloud if newer version exists.
Falls back to geometric placement if no model available.
This is the auto-improving loop in action.
"""

import torch
import numpy as np
from pathlib import Path
from annotation_ai import ArchitecturalPlacementModel
from model_manager import pull_latest_model

_model_instance = None
_device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH      = Path(__file__).parent.parent / "models" / "drafting_ai.pth"


def load_model():
    global _model_instance

    if _model_instance is not None:
        return _model_instance

    # Always check if cloud has a newer model
    pull_latest_model()

    if not MODEL_PATH.exists():
        print("No model available. Using geometric fallback.")
        return None

    try:
        model = ArchitecturalPlacementModel().to(_device)
        model.load_state_dict(
            torch.load(str(MODEL_PATH), map_location=_device)
        )
        model.eval()
        _model_instance = model
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Model load failed: {e}. Using geometric fallback.")
        return None


def predict_annotation_positions(wall_canvas, confidence_threshold=0.3):
    model = load_model()
    if model is None:
        return []

    try:
        tensor = torch.from_numpy(
            wall_canvas.astype(np.float32)
        ).unsqueeze(0).unsqueeze(0).to(_device)

        with torch.no_grad():
            heatmap = model(tensor)

        heatmap_np       = heatmap.squeeze().cpu().numpy()
        confident_pixels = np.argwhere(heatmap_np > confidence_threshold)
        return [[float(p[1]), float(p[0])] for p in confident_pixels]

    except Exception as e:
        print(f"Inference error: {e}.")
        return []


def canvas_to_mm_coordinates(canvas_coords, canvas_size=256, span_mm=50000.0):
    scale = span_mm / canvas_size
    return [[c[0] * scale, c[1] * scale] for c in canvas_coords]