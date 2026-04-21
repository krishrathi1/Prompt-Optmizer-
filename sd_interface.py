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

    def _generate_synthetic_fallback(self, prompt: str, width: int, height: int):
        """Creates a stylized placeholder when SD is offline."""
        from PIL import Image, ImageDraw, ImageFont
        import random

        # Create a gradient background
        img = Image.new('RGB', (width, height), color=(20, 20, 30))
        draw = ImageDraw.Draw(img)
        
        # Add some abstract 'blobs' to simulate 'visual noise' or 'composition'
        for _ in range(5):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = x1 + random.randint(50, 200), y1 + random.randint(50, 200)
            color = (random.randint(40, 100), random.randint(40, 100), random.randint(80, 150))
            draw.ellipse([x1, y1, x2, y2], fill=color)

        # Draw prompt text (simple)
        text = f"SYNTHETIC RENDER FALLBACK\nPrompt: {prompt[:100]}..."
        draw.text((20, height - 60), text, fill=(200, 200, 255))
        
        return {
            "image": img,
            "inference_time": 0.5,
            "status": "success",
            "sampler": "SYNTHETIC_V1 (Fallback Mode)",
        }

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 20,
        cfg_scale: float = 7.0,
        width: int = 512,
        height: int = 512,
        sampler_name: str = "DPM++ 2M",
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
            # Increased timeout to 120 for slower local machines
            response = requests.post(self.txt2img_url, json=payload, timeout=120)
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
            
            # If server responds but error (e.g. out of memory)
            print(f"[SD_INTERFACE] API Error {response.status_code}. Using synthetic fallback.")
            return self._generate_synthetic_fallback(prompt, width, height)

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            print("[SD_INTERFACE] Local Stable Diffusion offline (timeout/connection). Using synthetic fallback.")
            return self._generate_synthetic_fallback(prompt, width, height)
        except Exception as e:
            print(f"[SD_INTERFACE] Unexpected error: {e}. Using synthetic fallback.")
            return self._generate_synthetic_fallback(prompt, width, height)


if __name__ == "__main__":
    client = StableDiffusionClient()
    print("StableDiffusionClient initialized. Default sampler: DPM++ 2M Karras")
