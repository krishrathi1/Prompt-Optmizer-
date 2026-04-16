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

app = FastAPI(title="Prompt Optimizer PRO API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

optimizer = PromptOptimizer()
sd_client = StableDiffusionClient()
evaluator = PromptEvaluator()


class PromptRequest(BaseModel):
    prompt: str
    style: str = "Photoreal"
    use_ollama: bool = False


class GenerateRequest(BaseModel):
    original_prompt: str
    optimized_prompt: str
    negative_prompt: str = ""
    steps: int = 45
    cfg_scale: float = 10.0


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
        "clip_fallback": evaluator.fallback_mode,
        "nltk_path": str(nltk.data.path[0]) if hasattr(nltk, 'data') else "unknown",
        "personas": list(optimizer.expert_personas.keys()),
    }


@app.post("/api/optimize")
async def optimize_prompt(req: PromptRequest):
    try:
        logger.info(f"Optimizing prompt: {req.prompt[:50]}...")
        result = optimizer.optimize(req.prompt, style_preset=req.style, use_ollama=req.use_ollama)
        # Use jsonable_encoder to ensure everything is serializable
        return JSONResponse(
            content=jsonable_encoder(result),
            headers={"Content-Type": "application/json; charset=utf-8"}
        )
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        logger.error(f"Optimization Exception:\n{error_msg}")
        return JSONResponse(
            status_code=500,
            content={"error": "NLP Pipeline Error", "detail": str(e), "traceback": error_msg[:500]}
        )


@app.post("/api/generate")
async def generate_images(req: GenerateRequest):
    try:
        start_time = time.time()

        logger.info(f"Generating raw image: {req.original_prompt[:60]}")
        raw_res = sd_client.generate_image(req.original_prompt)

        logger.info(f"Generating optimized image: {req.optimized_prompt[:60]}")
        opt_res = sd_client.generate_image(
            req.optimized_prompt,
            negative_prompt=req.negative_prompt,
            steps=req.steps,
            cfg_scale=req.cfg_scale,
        )

        if raw_res['status'] != 'success' or opt_res['status'] != 'success':
            error_msg = raw_res.get('error', opt_res.get('error', 'Unknown SD error'))
            return JSONResponse(
                status_code=502,
                content={"error": f"Stable Diffusion offline: {error_msg}. Run with --api flag."},
            )

        raw_clip = evaluator.calculate_clip_score(raw_res['image'], req.original_prompt)
        opt_clip = evaluator.calculate_clip_score(opt_res['image'], req.optimized_prompt)
        aesthetic = evaluator.aesthetic_score_heuristic(opt_res['image'])
        complexity = min(len(req.optimized_prompt.split()) / 5, 10)
        composite = evaluator.calculate_composite_score(opt_clip, aesthetic, req.optimized_prompt)
        raw_aesthetic = evaluator.aesthetic_score_heuristic(raw_res['image'])
        raw_composite = evaluator.calculate_composite_score(raw_clip, raw_aesthetic, req.original_prompt)

        latency = round(time.time() - start_time, 2)
        logger.info(f"Done — composite: {composite} | latency: {latency}s")

        def pil_to_b64(img):
            buf = BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        return JSONResponse(content={
            "raw_image": pil_to_b64(raw_res['image']),
            "opt_image": pil_to_b64(opt_res['image']),
            "metrics": {
                "raw_clip": round(raw_clip * 10, 3),
                "opt_clip": round(opt_clip * 10, 3),
                "raw_aesthetic": raw_aesthetic,
                "aesthetic": aesthetic,
                "complexity": round(complexity, 2),
                "composite": composite,
                "raw_composite": raw_composite,
                "raw_tokens": len(req.original_prompt.split()),
                "opt_tokens": len(req.optimized_prompt.split()),
                "latency": latency,
            },
        })
    except Exception as e:
        import traceback
        logger.error(f"Generate Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


def find_free_port(start=8000, end=8100):
    """Find first available port in range."""
    import socket
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found in range {start}-{end}")


if __name__ == "__main__":
    port = find_free_port(8000, 8100)
    logger.info(f"Starting server on http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, reload=False)
