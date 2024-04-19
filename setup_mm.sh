apt-get update && apt-get install ffmpeg libsm6 libxext6 unar vim htop unzip gcc curl g++ -y

conda init bash && source ~/.bashrc &&
conda env create -f environment.yaml &&
conda activate tadp

pip install openmim &&
mim install mmcv-full==1.6.2 mmsegmentation==0.30.0

git clone https://github.com/damaggu/stable-diffusion
mv stable-diffusion stable_diffusion
cp -r stable_diffusion/ldm/ ./