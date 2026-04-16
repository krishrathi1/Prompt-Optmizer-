"""
server.py  —  Prompt Optimizer PRO  (v4.0)
==========================================
FastAPI backend with updated /api/generate using the new evaluate_full() suite.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import uvicorn
import os
import base64
from io import BytesIO
import time
import logging
import nltk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from optimizer_engine import PromptOptimizer
from sd_interface import StableDiffusionClient
from evaluator import PromptEvaluator

app = FastAPI(title="Prompt Optimizer PRO API — v4.0")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
static_dir  = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

optimizer  = PromptOptimizer()
sd_client  = StableDiffusionClient()
evaluator  = PromptEvaluator()


# ─────────────────────────────────────────────────────────────────────────────
#  Pydantic Models
# ─────────────────────────────────────────────────────────────────────────────

class PromptRequest(BaseModel):
    prompt: str
    style: str = "Photoreal"
    use_ollama: bool = False


class GenerateRequest(BaseModel):
    original_prompt:  str
    optimized_prompt: str
    negative_prompt:  str   = ""
    steps:            int   = 45
    cfg_scale:        float = 8.0


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        return templates.TemplateResponse(request=request, name="index.html", context={})
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"UI Loading Error:\n{error_trace}")
        return HTMLResponse(
            content=f"<h1>Error loading UI</h1><pre>{error_trace}</pre>",
            status_code=500,
        )


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "4.0",
        "clip_fallback": evaluator.fallback_mode,
        "sts_available": evaluator.sts_model is not None,
        "lm_available": evaluator._lm_available,
        "nltk_path": str(nltk.data.path[0]) if hasattr(nltk, 'data') else "unknown",
        "personas": list(optimizer.expert_personas.keys()),
    }


@app.post("/api/optimize")
async def optimize_prompt(req: PromptRequest):
    try:
        logger.info(f"[/api/optimize] Prompt: {req.prompt[:60]}...")
        result = optimizer.optimize(
            req.prompt,
            style_preset=req.style,
            use_ollama=req.use_ollama
        )
        return JSONResponse(
            content=jsonable_encoder(result),
            headers={"Content-Type": "application/json; charset=utf-8"}
        )
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        logger.error(f"[/api/optimize] Exception:\n{error_msg}")
        return JSONResponse(
            status_code=500,
            content={"error": "NLP Pipeline Error", "detail": str(e), "traceback": error_msg[:500]}
        )


@app.post("/api/generate")
async def generate_images(req: GenerateRequest):
    try:
        start_time = time.time()

        logger.info(f"[/api/generate] Raw: {req.original_prompt[:60]}")
        raw_res = sd_client.generate_image(req.original_prompt)

        logger.info(f"[/api/generate] Opt: {req.optimized_prompt[:60]}")
        opt_res = sd_client.generate_image(
            req.optimized_prompt,
            negative_prompt=req.negative_prompt,
            steps=req.steps,
            cfg_scale=req.cfg_scale,
            sampler_name="DPM++ 2M Karras",   # W15 fix
        )

        if raw_res['status'] != 'success' or opt_res['status'] != 'success':
            error_msg = raw_res.get('error', opt_res.get('error', 'Unknown SD error'))
            return JSONResponse(
                status_code=502,
                content={"error": f"Stable Diffusion offline: {error_msg}. Run with --api flag."},
            )

        inference_time = round(time.time() - start_time, 2)

        # Full evaluation suite (v3.0)
        eval_report = evaluator.evaluate_full(
            original_prompt=req.original_prompt,
            optimized_prompt=req.optimized_prompt,
            raw_image=raw_res['image'],
            opt_image=opt_res['image'],
            inference_time=inference_time,
        )

        def pil_to_b64(img):
            buf = BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        logger.info(
            f"[/api/generate] Done — "
            f"composite improvement: {eval_report['composite']['improvement']} | "
            f"latency: {inference_time}s"
        )

        return JSONResponse(content={
            "raw_image": pil_to_b64(raw_res['image']),
            "opt_image": pil_to_b64(opt_res['image']),
            "evaluation": eval_report,
            "latency": inference_time,
        })

    except Exception as e:
        import traceback
        logger.error(f"[/api/generate] Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluate_text")
async def evaluate_text_only(req: PromptRequest):
    """
    Text-only evaluation endpoint — no SD required.
    Returns full NLP metric suite for original vs optimized text.
    """
    try:
        opt_result = optimizer.optimize(req.prompt, style_preset=req.style)
        eval_report = evaluator.evaluate_full(
            original_prompt=req.prompt,
            optimized_prompt=opt_result["optimized_prompt"],
        )
        return JSONResponse(content=jsonable_encoder({
            "original": req.prompt,
            "optimized": opt_result["optimized_prompt"],
            "optimization": opt_result,
            "evaluation": eval_report,
        }))
    except Exception as e:
        import traceback
        logger.error(f"[/api/evaluate_text] Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Port discovery & launch
# ─────────────────────────────────────────────────────────────────────────────

def find_free_port(start: int = 8000, end: int = 8100) -> int:
    import socket
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port in {start}-{end}")


if __name__ == "__main__":
    port = find_free_port(8000, 8100)
    logger.info(f"Starting Prompt Optimizer PRO v4.0 at http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, reload=False)
