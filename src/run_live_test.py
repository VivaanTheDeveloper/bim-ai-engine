import os
import numpy as np
from pathlib import Path

# Import elements across code modules
from geometry_engine import extract_geometry
from schedule_engine import compile_component_schedules
from preprocessor import ifc_to_tensor
from inference_engine import predict_annotation_positions, canvas_to_mm_coordinates
from collision_engine import resolve_annotation_collisions
from cad_compiler import compile_production_gfc_dxf

def execute_unseen_test(ifc_test_file):
    print("🚀 Initializing live automation inference test sequence...")
    Path("output").mkdir(exist_ok=True)
    
    out_dxf = "output/Inference_Output_Blueprint.dxf"
    out_csv = "output/Inference_Output_Schedule.csv"
    
    # 1. Math Slicing Layer
    print("Slicing 3D architecture into 2D linear paths...")
    geometry_package = extract_geometry(ifc_test_file)
    
    # 2. Automated Excel Schedule Matrix Extraction
    print("Compiling project schedules...")
    compile_component_schedules(ifc_test_file, out_csv)
    
    # 3. Convert input model to tensor for your AI script
    wall_canvas = ifc_to_tensor(ifc_test_file).squeeze(0) # Strip batch channel for processing
    
    # 4. Predict dimension placements using your trained model weights file
    print("Running geometry matrix through trained model layers...")
    raw_pixels = predict_annotation_positions(wall_canvas, confidence_threshold=0.3)
    print(f"AI targeted {len(raw_pixels)} coordinates for annotation placement.")
    
    # 5. Convert pixel indices back to structural millimeters
    real_world_mm_coords = canvas_to_mm_coordinates(raw_pixels, canvas_size=256, real_world_span_mm=50000.0)
    
    # 6. Apply collision filters to prevent text overlap
    formatted_walls = [((w.x1, w.y1), (w.x2, w.y2)) for w in geometry_package.walls]
    resolved_coords = resolve_annotation_collisions(formatted_walls, real_world_mm_coords)
    
    # Map predictions to matching CAD lines
    final_cad_dimensions = []
    for idx, label_pos in enumerate(resolved_coords):
        if idx < len(geometry_package.walls):
            w = geometry_package.walls[idx]
            final_cad_dimensions.append({
                'p1': [w.x1, w.y1],
                'p2': [w.x2, w.y2],
                'placement': label_pos
            })
            
    # 7. Package and write the output to a native DXF file
    print("Assembling native multi-layer drawing CAD sheets...")
    compile_production_gfc_dxf(formatted_walls, final_cad_dimensions, out_dxf)
    print("\n🎉 Verification process finished. Output files ready in output/ directory.")

if __name__ == "__main__":
    execute_unseen_test("test_unseen_structure.ifc")