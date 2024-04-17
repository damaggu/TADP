conda init bash && source ~/.bashrc &&
conda env create -f environment.yaml &&
conda activate tadp

#pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install openmim &&
mim install mmcv-full==1.6.2 mmsegmentation==0.30.0
#pip install pytorch_lightning==1.6.5