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

wget -P ./data/ http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf ./data/VOCtrainval_11-May-2012.tar -C ./data/

wget -O ./data/SegmentationClassAug.zip https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0
unzip ./data/SegmentationClassAug.zip "SegmentationClassAug/*" -d ./data

rm -rf ./data/VOCdevkit/VOC2012/SegmentationClass
mv ./data/SegmentationClassAug ./data/VOCdevkit/VOC2012/SegmentationClass

cp trainaug.txt ./data/VOCdevkit/VOC2012/ImageSets/Segmentation/


# for obj detection
#pip install openmim &&
#mim install "mmcv>=2.0.0"



# Download checkpoints
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt - O checkpoints/v1-5-pruned-emaonly.ckpt &&
wget "https://www.dropbox.com/scl/fi/ox7tvyhsoqyhkkrf4z47b/tadp_depth_blipmin40.ckpt?rlkey=rocsl40cdia8mu28culdrjoo3&dl=1" -O checkpoints/tadp_depth_blipmin40.ckpt &&
wget "https://www.dropbox.com/scl/fi/men4sn4khyht5i1h6cdyh/tadp_seg_blipmin40.ckpt?rlkey=onlpos0js4g3wsm82ycku0rwd&dl=1" -O checkpoints/tadp_seg_blipmin40.ckpt


pip install gdown

git clone https://github.com/naoto0804/cross-domain-detection.git
mv cross-domain-detection data/
bash data/cross-domain-detection/datasets/prepare.sh
mv watercolor data/cross-domain-detection/datasets/
mv comic data/cross-domain-detection/datasets/
mv clipart data/cross-domain-detection/datasets/