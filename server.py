import os
import tempfile
import logging
from typing import List, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

from PIL import Image, ImageSequence
import numpy as np
import cv2

# Lazy import so the app can start even if paddle isn't ready yet
from paddleocr import PaddleOCR  # type: ignore

# Thread-safety: Paddle isn't guaranteed thread-safe across threads;
# use a single shared instance + simple mutex.
import threading
_ocr_lock = threading.Lock()
_init_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Warm start route (optional): downloads models at boot so first request is snappy
    try:
        ocr = _init_ocr()
        # tiny white image to trigger model load
        img = Image.new("RGB", (32, 32), "white")
        _ = ocr.predict(_img_to_bgr_array(img))
        logging.getLogger("uvicorn").info("PaddleOCR warmed up.")
    except Exception as e:
        # Don't crash—/ocr will return 555 when invoked
        logging.getLogger("uvicorn").warning(f"Paddle warmup failed: {e}")
    
    yield
    # Shutdown: Nothing to do on shutdown for now
    

app = FastAPI(title="PaddleOCR TIFF Server", version="1.0", lifespan=lifespan)

# ── Config ────────────────────────────────────────────────────────────────────
CUSTOM_PADDLE_STATUS = 699  # non-standard status for "Paddle-OCR error"

# PaddleOCR is heavy. Create once and reuse. Guard with a flag.
_ocr = None
_initialized = False

# ── Utilities ─────────────────────────────────────────────────────────────────
def _init_ocr() -> PaddleOCR:
    global _ocr, _initialized
    if _initialized and _ocr is not None:
        return _ocr
    with _init_lock:
        if not _initialized or _ocr is None:
            _ocr = PaddleOCR(lang='en')  # set use_gpu as needed
            _initialized = True
    return _ocr


def _img_to_bgr_array(img: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image (any mode) to OpenCV-style BGR numpy array.
    """
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        # Convert grayscale to BGR to keep codepath uniform
        img = img.convert("RGB")
    arr = np.array(img)
    # PIL gives RGB; Paddle (via OpenCV) expects BGR
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


def _preprocess_for_ocr(img: Image.Image) -> np.ndarray:
    """
    Optional preprocessing: upscale lightly and apply adaptive threshold on very low-contrast pages.
    We keep it mild to avoid hurting quality on good scans.
    """
    # Light upscale for tiny scans (max dim ~2000px)
    max_dim = max(img.size)
    if max_dim < 1500:
        scale = min(2000 / max_dim, 2.0)
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    bgr = _img_to_bgr_array(img)

    # Heuristic: if very low variance, try adaptive threshold on luminance
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.std() < 25:
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 15
        )
        bgr = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

    return bgr


def _ocr_tiff_pages(fp: str) -> Tuple[str, float]:
    """
    Run OCR across all frames of a TIFF, return joined text and score tuple.
    """
    ocr = _init_ocr()
    
    texts = []
    scores = []

    with Image.open(fp) as im:
        for _, frame in enumerate(ImageSequence.Iterator(im), start=1):
            bgr = _preprocess_for_ocr(frame)
            # Paddle can take ndarray directly
            with _ocr_lock:
                result = ocr.predict(bgr)

            # Extract text from results
            for res in result:
                if hasattr(res, 'json') and res.json:
                    json_data = res.json
                    if 'res' in json_data:
                        res_data = json_data['res']

                        if "rec_texts" in res_data:
                            page_texts = res_data['rec_texts']
                            for text in page_texts:
                                if text:
                                    texts.append(text)
                        if 'rec_scores' in res_data:
                            page_sccores = res_data['rec_scores']
                            scores.extend(page_sccores)

                    if 'res' in json_data and 'rec_texts' in json_data['res']:
                        for text in json_data['res']['rec_texts']:
                            if text:
                                texts.append(text)

        # Join pages with clear separators
        # You asked for a single {text: "..."} field, so we flatten:
        combined = "\n\n".join(
            t for t in texts if t
        ).strip()
        
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return combined, avg_score


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"


@app.post("/ocr")
async def ocr_tiff(file: UploadFile = File(...)):
    # Validate content type / extension
    filename = (file.filename or "").lower()
    ctype = (file.content_type or "").lower()
    if not (filename.endswith(".tif") or filename.endswith(".tiff") or "tiff" in ctype):
        raise HTTPException(status_code=415, detail="Please upload a .tiff/.tif file")

    # Persist to a temp file so PIL can seek multipage frames reliably
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tiff") as tmp:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to read upload: {e}") from e
    finally:
        await file.close()

    try:
        # Run the OCR work in a thread to avoid blocking the event loop
        text, score = await run_in_threadpool(_ocr_tiff_pages, tmp_path)
        return JSONResponse({"text": text, "score": score})
    except RuntimeError as e:
        # Paddle init or inference error → custom status
        return JSONResponse({"error": str(e)}, status_code=CUSTOM_PADDLE_STATUS)
    except Exception as e:
        # Non-Paddle issues (bad image, corrupt tiff, etc.)
        raise HTTPException(status_code=422, detail=f"Failed to process TIFF: {e}") from e
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
        workers=int(os.environ.get("WORKERS", "1")),
    )
