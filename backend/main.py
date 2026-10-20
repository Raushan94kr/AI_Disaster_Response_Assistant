from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile
import shutil

from backend.services.analysis_service import analyze_xbd_image

app = FastAPI(
    title="AI Disaster Response Assistant API",
    description="Backend API for xBD building-level disaster damage assessment.",
    version="1.0.0",
)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AI Disaster Response Assistant"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename supplied.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are supported.",
        )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="xbd_upload_") as temp:
            temp_path = Path(temp.name)
            shutil.copyfileobj(file.file, temp)

        result = analyze_xbd_image(temp_path, file.filename)
        return JSONResponse(content=result)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
