import requests
import io
import base64
from PIL import Image
import time

class StableDiffusionClient:
    def __init__(self, base_url="http://127.0.0.1:7860"):
        self.base_url = base_url
        self.txt2img_url = f"{base_url}/sdapi/v1/txt2img"
        
    def generate_image(self, prompt, negative_prompt="", steps=35, cfg_scale=9.0, width=512, height=512):
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "sampler_name": "Euler a"
        }
        
        start_time = time.time()
        try:
            response = requests.post(self.txt2img_url, json=payload, timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                r = response.json()
                image_data = base64.b64decode(r['images'][0])
                image = Image.open(io.BytesIO(image_data))
                
                return {
                    "image": image,
                    "inference_time": end_time - start_time,
                    "status": "success"
                }
            else:
                return {
                    "status": "error",
                    "error": f"API Error: {response.status_code}",
                    "inference_time": 0
                }
        except requests.exceptions.ConnectionError:
            return {
                "status": "error",
                "error": "Connection Refused. Is Stable Diffusion running with --api?",
                "inference_time": 0
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "inference_time": 0
            }

if __name__ == "__main__":
    client = StableDiffusionClient()
    # Replace with a real prompt if testing with a running SD instance
    print("Stable Diffusion Client initialized.")
