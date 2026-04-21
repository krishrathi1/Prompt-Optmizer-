"""
sd_interface.py — Stable Diffusion API Client (v1.1)
=====================================================
W15 fix: DPM++ 2M Karras as default sampler (was "Euler a").
"""

import requests
import io
import base64
from PIL import Image
import time


class StableDiffusionClient:
    def __init__(self, base_url: str = "http://127.0.0.1:7860"):
        self.set_base_url(base_url)

    def set_base_url(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.txt2img_url = f"{self.base_url}/sdapi/v1/txt2img"
        self.options_url = f"{self.base_url}/sdapi/v1/options"

    def check_health(self, timeout: int = 5):
        try:
            response = requests.get(self.options_url, timeout=timeout)
            if response.status_code == 200:
                return {"ok": True, "base_url": self.base_url, "error": None}
            return {
                "ok": False,
                "base_url": self.base_url,
                "error": f"SD API status {response.status_code}",
            }
        except requests.exceptions.ConnectionError:
            return {
                "ok": False,
                "base_url": self.base_url,
                "error": "Connection refused. Is Stable Diffusion running with --api?",
            }
        except Exception as e:
            return {"ok": False, "base_url": self.base_url, "error": str(e)}

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 35,
        cfg_scale: float = 9.0,
        width: int = 512,
        height: int = 512,
        sampler_name: str = "DPM++ 2M Karras",   # W15 fix: was "Euler a"
    ):
        payload = {
            "prompt":          prompt,
            "negative_prompt": negative_prompt,
            "steps":           steps,
            "cfg_scale":       cfg_scale,
            "width":           width,
            "height":          height,
            "sampler_name":    sampler_name,
        }

        start_time = time.time()
        try:
            response = requests.post(self.txt2img_url, json=payload, timeout=60)
            end_time = time.time()

            if response.status_code == 200:
                r = response.json()
                image_data = base64.b64decode(r['images'][0])
                image = Image.open(io.BytesIO(image_data))
                return {
                    "image": image,
                    "inference_time": round(end_time - start_time, 2),
                    "status": "success",
                    "sampler": sampler_name,
                }
            return {
                "status": "error",
                "error": f"SD API status {response.status_code}",
                "inference_time": 0,
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": "error",
                "error": "Connection refused. Is Stable Diffusion running with --api?",
                "inference_time": 0,
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "inference_time": 0}


if __name__ == "__main__":
    client = StableDiffusionClient()
    print("StableDiffusionClient initialized. Default sampler: DPM++ 2M Karras")
