apt-get update && apt-get install ffmpeg libsm6 libxext6 unar vim htop unzip gcc curl g++ -y

# detectron2
git clone https:// github.com/facebookresearch/detectron2.git
mv detectron2 detectron2_repo
cp -r detectron2_repo/detectron2/ ./
rm -rf detectron2_repo
pip install fvcore pycocotools cloudpickle chainercv

pip install git+https://github.com/huggingface/diffusers.git@2764db3194fc1b5069df7292fd938657d8568995
pip install torchvision==0.15.2


pip install torch
pip install -r requirements.txt

#git clone https://github.com/wl-zhao/VPD
git clone https://github.com/damaggu/stable-diffusion
mv stable-diffusion stable_diffusion
cp -r stable_diffusion/ldm/ ./

export PYTHONPATH=$PYTHONPATH:$(pwd)
python create_class_embeddings.py --dataset pascal
