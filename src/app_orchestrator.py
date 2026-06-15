"""
app_orchestrator.py
-------------------
Enterprise FastAPI backend.
Every request is authenticated with a firm's API key.
Unlimited processing per key. No credit limits.
"""
import pathlib
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from geometry_engine import extract_geometry
from schedule_engine import compile_component_schedules
from collision_engine import resolve_annotation_collisions
from cad_compiler import compile_production_gfc_dxf
from inference_engine import predict_annotation_positions, canvas_to_mm_coordinates
from preprocessor import ifc_to_tensor
from api_key_manager import validate_api_key, log_job

app = FastAPI(
    title="BIM AI Engine API",
    description="Enterprise IFC to DXF conversion powered by AI",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "online", "version": "2.0.0"}


@app.post("/process-bim-model")
async def process_bim_model(
    file: UploadFile = File(...),
    x_api_key: str = Header(None)
):
    # ── Authenticate the firm ─────────────────────────────────────────────
    firm = validate_api_key(x_api_key)
    if not firm:
        raise HTTPException(
            status_code=401,
            detail="Invalid or inactive API key. Contact vivaan@bim-ai.com"
        )

    firm_name = firm["firm_name"]
    print(f"\n[REQUEST] {firm_name} → {file.filename}")

    # ── File paths ────────────────────────────────────────────────────────
    safe_name       = file.filename.replace(" ", "_")
    base_name       = safe_name.split(".")[0]
    DATA_DIR = os.getenv("DATA_DIR", str(pathlib.Path.home() / "BIM_AI_Engine_Data"))

    temp_ifc_path   = f"{DATA_DIR}/dataset/raw_ifc/temp_{safe_name}"
    output_dxf_path = f"{DATA_DIR}/output/GFC_{firm_name.replace(' ','_')}_{base_name}.dxf"
    output_csv_path = f"{DATA_DIR}/output/SCHEDULE_{firm_name.replace(' ','_')}_{base_name}.csv"

    os.makedirs(f"{DATA_DIR}/output", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/dataset/raw_ifc", exist_ok=True)

    os.makedirs("../output", exist_ok=True)
    os.makedirs("../dataset/raw_ifc", exist_ok=True)

    with open(temp_ifc_path, "wb") as f:
        f.write(await file.read())

    try:
        # ── Step 1: Extract geometry ──────────────────────────────────────
        print(f"[1/5] Extracting geometry...")
        pkg = extract_geometry(temp_ifc_path)

        if pkg.is_empty:
            return {"status": "Failed", "error": "No walls found in IFC file."}

        wall_vectors = [[[w.x1, w.y1], [w.x2, w.y2]] for w in pkg.walls]
        print(f"[1/5] {len(wall_vectors)} walls extracted.")

        # ── Step 2: Door schedule ─────────────────────────────────────────
        print(f"[2/5] Extracting door schedule...")
        try:
            compile_component_schedules(temp_ifc_path, output_csv_path)
        except Exception as e:
            print(f"[2/5] Schedule warning: {e}")

        # ── Step 3: AI inference ──────────────────────────────────────────
        print(f"[3/5] Running AI inference...")
        ai_placements = []
        try:
            wall_canvas      = ifc_to_tensor(temp_ifc_path).squeeze()
            canvas_preds     = predict_annotation_positions(wall_canvas)
            ai_placements    = canvas_to_mm_coordinates(canvas_preds)
        except Exception as e:
            print(f"[3/5] Inference warning: {e}")

        if not ai_placements:
            ai_placements = [[seg[0][0], seg[0][1] + 800] for seg in wall_vectors]

        # ── Step 4: Collision avoidance ───────────────────────────────────
        print(f"[4/5] Collision avoidance...")
        try:
            safe_points = resolve_annotation_collisions(wall_vectors, ai_placements)
        except Exception:
            safe_points = ai_placements

        # ── Step 5: Compile DXF ───────────────────────────────────────────
        print(f"[5/5] Compiling DXF...")
        max_dims = min(len(wall_vectors), len(safe_points))
        dims = [
            {"p1": wall_vectors[i][0], "p2": wall_vectors[i][1], "placement": safe_points[i]}
            for i in range(max_dims)
        ]
        compile_production_gfc_dxf(wall_vectors, dims, output_dxf_path)

        # ── Log to Supabase ───────────────────────────────────────────────
        try:
            log_job(x_api_key, firm_name, file.filename,
                    len(wall_vectors), output_dxf_path, output_csv_path)
        except Exception:
            pass

        print(f"[SUCCESS] {firm_name} → {len(wall_vectors)} walls → files ready.")

        return {
            "status":        "Success",
            "firm":          firm_name,
            "dxf_blueprint": output_dxf_path,
            "csv_schedule":  output_csv_path,
            "wall_count":    len(wall_vectors),
            "message":       f"Processed {len(wall_vectors)} walls successfully."
        }

    except Exception as e:
        print(f"[CRASH] {e}")
        return {"status": "Failed", "error_log": str(e)}

    finally:
        if os.path.exists(temp_ifc_path):
            os.remove(temp_ifc_path)


@app.get("/output/{filename}")
async def download_file(filename: str, x_api_key: str = Header(None)):
    firm = validate_api_key(x_api_key)
    if not firm:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    file_path = f"../output/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app_orchestrator:app", host="0.0.0.0", port=port, reload=False)