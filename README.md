# Text-image Alignment for Diffusion-based Perception (TADP)

[![Project Page](https://img.shields.io/badge/Project%20Page-Link)](https://www.vision.caltech.edu/tadp/)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2310.00031)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uadQ8iukjmnjEScn1qKWuy9ijcUbPMuN?usp=sharing)
<!-- [![Open In Colab](doc/badges/badge-colab.svg)](https://colab.research.google.com/drive/...) -->
<!-- [![Hugging Face (LCM) Space](https://img.shields.io/badge/🤗%20Hugging%20Face(LCM)-Space-yellow)](https://huggingface.co/spaces/...) -->
<!--[![Hugging Face (LCM) Model](https://img.shields.io/badge/🤗%20Hugging%20Face(LCM)-Model-green)](https://huggingface.co/prs-eth/marigold-lcm-v1-0) -->
<!-- [![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green)](https://huggingface.co/...) -->
<!-- [![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0) -->
<!-- [![Website](https://img.shields.io/badge/Project-Website-1081c2)](https://arxiv.org/abs/2312.02145) -->
<!-- [![GitHub](https://img.shields.io/github/stars/prs-eth/Marigold?style=default&label=GitHub%20★&logo=github)](https://github.com/...) -->
<!-- [![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)]() -->
<!-- [![Docker](doc/badges/badge-docker.svg)]() -->


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/text-image-alignment-for-diffusion-based/semantic-segmentation-on-nighttime-driving)](https://paperswithcode.com/sota/semantic-segmentation-on-nighttime-driving?p=text-image-alignment-for-diffusion-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/text-image-alignment-for-diffusion-based/weakly-supervised-object-detection-on-comic2k)](https://paperswithcode.com/sota/weakly-supervised-object-detection-on-comic2k?p=text-image-alignment-for-diffusion-based)	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/text-image-alignment-for-diffusion-based/semantic-segmentation-on-pascal-voc-2012-val)](https://paperswithcode.com/sota/semantic-segmentation-on-pascal-voc-2012-val?p=text-image-alignment-for-diffusion-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/text-image-alignment-for-diffusion-based/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=text-image-alignment-for-diffusion-based)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/text-image-alignment-for-diffusion-based/semantic-segmentation-on-ade20k)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k?p=text-image-alignment-for-diffusion-based)


---

Official implementation of the paper **Text-Image Alignment for Diffusion-based Perception (CVPR 2024)**.


[Neehar Kondapaneni*](https://nkondapa.github.io/),
[Markus Marks*](https://damaggu.github.io/),
[Manuel Knott*](https://scholar.google.com/citations?user=e9xfiKEAAAAJ&hl=en),
[Rogerio Guimaraes](https://rogeriojr.com/),
[Pietro Perona](https://www.vision.caltech.edu/)

![methods](assets/methods.gif)


## Setup

We have 2 seperate shell scripts for setting up the environment. 

- `setup.sh` for setting up the environment for Pascal VOC Semantic Segmentation and Watercolor2k and Comic2k Object Detection.
- `setup_mm.sh` for setting up the environment for ADE20k Semantic Segmentation, NYUv2 Depth Estimation, Nighttime Driving, and Dark Zurich Semantic Segmentation (using MM libraries).

```bash
bash setup.sh
```

## Inference

If you want to use our models for inference, there are two options available:

### Single image inference
We provide a simple interface to load our model checkpoints and run inference with custom image and text inputs.
Please refer to the [demo/](demo/) directory for examples.

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
python demo/depth_inference.py
python demo/seg_inference.py
python demo/detection_inference.py
python demo/seg_inference_driving.py
```

### Whole data set testing
If you want to generate results for a whole dataset that was used in our study (e.g., ADE20k, NYUv2) using pre-generated captions, 
please refer to the [test_tadp_mm.py](test_tadp_mm.py) and [test_tadp_depth.py](test_tadp_depth.py) scripts.


## Training

TODO

## Experiments

All results that are reported in our paper can be reproduced using the scripts in the [cvpr_experiments/](cvpr_experiments/) directory.

# Acknowledgements
This code is based on [VPD](https://github.com/wl-zhao/VPD), [diffusers](https://github.com/wl-zhao/VPD), [stable-diffusion](https://github.com/CompVis/stable-diffusion), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [LAVT](https://github.com/yz93/LAVT-RIS), and [MIM-Depth-Estimation](https://github.com/SwinTransformer/MIM-Depth-Estimation).

# Citation
```
@article{kondapaneni2023tadp,
  title={Text-image Alignment for Diffusion-based Perception},
  author={Kondapaneni, Neehar and Marks, Markus and Knott, Manuel and Guimaraes, Rogerio and Perona, Pietro},
  journal={arXiv preprint arXiv:2310.00031},
  year={2023}
}
```

