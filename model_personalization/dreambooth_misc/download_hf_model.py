
# hf path
# path = './dreambooth_models/watercolor_v1/dreambooth_watercolor_5/checkpoint-1000/unet/diffusion_pytorch_model.bin'
path = './dreambooth_models/checkpoint-1000/unet/'

# load the model
from diffusers import DiffusionPipeline, UNet2DConditionModel

# model = UNet2DConditionModel.from_pretrained(path)




import os
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline

ckpt = "v1-5-pruned.ckpt"
# ckpt = "v1-5-pruned-emaonly.ckpt"
repo = "runwayml/stable-diffusion-v1-5"
# repo = "CompVis/stable-diffusion-v1-5"
out_dir = "checkpoints"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(os.path.join(out_dir, ckpt)):
    # hf_hub_download(repo_id="runwayml/stable-diffusion-v1-5", filename=ckpt, local_dir=out_dir)
    # hf_hub_download(repo_id=repo, local_dir=out_dir)
    DiffusionPipeline.from_pretrained(repo, cache_dir=out_dir, force_download=True)
