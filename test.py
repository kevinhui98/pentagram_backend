import os
import modal
import io
from fastapi import Response , HTTPException, Query,Request
import requests
from datetime import datetime, timezone

app = modal.App("pentagram")
image = modal.Image.debian_slim().pip_install("diffusers","transformers[sentencepiece]","accelerate","fastapi[standard]",)
# .env({
#     "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments=True" # backfiring
# })

with image.imports():
    # from diffusers import FluxPipeline
    from diffusers import DiffusionPipeline,AutoPipelineForText2Image,FluxPipeline
    import torch
    import io
    import os
    from fastapi import Response , HTTPException, Query,Request
    

@app.cls(image=image, gpu="A100-80GB",secrets=[modal.Secret.from_name('huggingface-secret'),modal.Secret.from_name('API-KEY')],container_idle_timeout = 300)

class Model:
    @modal.build()
    @modal.enter()
    def load_weights(self):
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"])
        print("PyTorch version:", torch.__version__)
        # self.pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",torch_dtype=torch.float16)
        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        # self.pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16)
        # self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16)
        # self.pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        self.pipe.enable_model_cpu_offload()
        # self.pipe.to("cuda")
        self.api_key = os.environ["API_KEY"]
        print(os.environ["API_KEY"])
    @modal.web_endpoint()
    def generate(self, request:Request, prompt: str = Query(...,description="The prompt for image generation")):
        api_key =request.headers.get("X-API-KEY")
        if api_key != self.api_key:
            raise HTTPException(
                status_code = 401,
                detail="Unauthorized"
            )
        image = self.pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        buffer = io.BytesIO()
        image.save(buffer,format="JPEG")
        return Response(content=buffer.getvalue(),media_type="image/jpeg")
@app.function(image=image,schedule=modal.Cron("*/5 * * * *"),secrets=[modal.Secret.from_name('API-KEY')])  # run at the start of the hour
def update_keep_warm():
    import os
    import requests
    from datetime import datetime, timezone
    

    health_url = "https://kevinhui98--pentagram-model-health.modal.run"
    generate_url = "https://kevinhui98--pentagram-model-generate.modal.run"

    try:
        health_response = requests.get(health_url)
        health_response.raise_for_status()
        print(f"Health check at {health_response.json().get('timestamp', 'N/A')}")
    except requests.RequestException as e:
        print(f"Health check failed: {e}")

    try:
        headers = {"X-API-KEY": os.environ["API_KEY"]}
        generate_response = requests.get(generate_url, headers=headers)
        generate_response.raise_for_status()
        print(f"Generate endpoint tested successfully at: {datetime.now(timezone.utc).isoformat()}")
    except requests.RequestException as e:
        print(f"Generate endpoint test failed: {e}")