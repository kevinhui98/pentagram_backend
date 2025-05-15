import os
import modal
import io
from fastapi import Response , HTTPException, Query,Request
from datetime import datetime, timezone
import requests

image = modal.Image.debian_slim().pip_install("diffusers","transformers","accelerate","fastapi[standard]","huggingface_hub").env({
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments=True"  # ✅ Safe and supported in 2.7
})

app = modal.App(name="Pentagram", image=image)

with image.imports():
    # from fastapi import Response
    import torch
    # from diffusers import DiffusionPipeline
    import os
    import modal
    import io
    # from diffusers import AutoPipelineForText2Image
    # from diffusers import FluxPipeline
    from diffusers import DiffusionPipeline
    # from diffusers import StableDiffusion3Pipeline
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
    

@app.cls(secrets=[modal.Secret.from_name('huggingface-secret'),modal.Secret.from_name("API-KEY")],gpu="A10G",container_idle_timeout = 300)
# @app.cls(secrets=[modal.Secret.from_name("black_forest_labs")],gpu="A10G")
class Model:
    @modal.build()
    @modal.enter()
    def load_weights(self):
        # self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",torch_dtype=torch.bfloat16)
        # self.pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
        # self.pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16)
        self.pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",torch_dtype=torch.bfloat16)
        # self.pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
        self.pipe.to("cuda")
        self.api_key = os.environ["API_KEY"]
        print(self.api_key)
    # @app.function
    @modal.web_endpoint()
    def generate(self, request:Request, prompt: str ="A cinematic shot of a baby racoon wearing an intricate italian priest robe"):  
        # api_key =request.headers.get("X-API-KEY")
        # print(api_key)
        # if api_key != self.api_key:
        #     raise HTTPException(
        #         status_code = 401,
        #         detail="Unauthorized"
        #     )
        image = self.pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        # print("image",image)
        return Response(content = buffer.getvalue(),media_type="image/jpeg")
    
# @app.function(schedule=modal.Cron("*/5 * * * *"),secrets=[modal.Secret.from_name('API-KEY')])  # run at the start of the hour
# def update_keep_warm():
#     health_url = "https://kevinhui98--pentagram-model-health.modal.run"
#     generate_url = "https://kevinhui98--pentagram-model-generate.modal.run"

#     health_response = requests.get(health_url)
#     print(f"Health check at {health_response.json()['timestamp']}")

#     headers = {"X-API-KEY":os.environ["API_KEY"]}
#     generate_response = requests.get(generate_url,headers=headers)
#     print(f"Generate endpoint tested successfully at: {datetime.now(timezone.utc).isoformat()}")