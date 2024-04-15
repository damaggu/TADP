conda init bash && source ~/.bashrc &&
conda env create -f stable-diffusion/environment.yaml &&
conda activate ldm

pip install openmim &&
mim install mmcv-full==1.6.2 mmsegmentation==0.30.0
pip install timm wandb
pip install pytorch_lightning==1.6.5