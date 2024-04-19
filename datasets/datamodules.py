import cv2
import os
import torch

import numpy as np
from PIL import Image
from torchvision import transforms

from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

CLASS_NAMES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]


# adapted
# from https://github.com/vidit09/domaingen/blob/main/data/datasets/pascal_voc_adaptation.py
def load_voc_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load Pascal VOC detection annotations to Detectron2 format.
    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str_)

    # Needs to read many small annotation files. Makes sense at local
    # annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    annotation_dirname = os.path.join(dirname, "Annotations/")
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        with open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls not in CLASS_NAMES:
                continue
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)

            bbox[0] -= 1.0
            bbox[1] -= 1.0

            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


class PascalVOCDataModule(Dataset):
    def __init__(self, data_dir, split, class_names, image_size=(224, 224), dataset_name="PascalVOC"):
        self.data_dir = data_dir
        self.split = split
        self.class_names = class_names

        self.data = load_voc_instances(self.data_dir, self.split, self.class_names)

        self.target_width = image_size[0]
        self.target_height = image_size[1]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.data)

    def get_dataset_name(self):
        return self.dataset_name

    def __getitem__(self, idx):
        ret = self.data[idx]
        image = Image.open(ret["file_name"]).convert("RGB")
        orig_width, orig_height = image.size
        bboxes = [x["bbox"] for x in ret["annotations"]]

        if self.transform:
            image = self.transform(image)

        bboxes = torch.tensor(bboxes, dtype=torch.float32)

        metas = {
            'image_id': ret["image_id"],
            'image_path': ret["file_name"],
        }
        return image, bboxes, torch.tensor([x["category_id"] for x in ret["annotations"]]), metas

    def get_image(self, idx, use_pil=True, use_cv2=False, return_img_file=False, return_class_label=False):
        item = self.data[idx]
        img_file = item["file_name"]
        if use_cv2:
            image = cv2.imread(img_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif use_pil:
            image = Image.open(img_file).convert("RGB")

        ret_list = [image]

        if return_img_file:
            ret_list.append(img_file)

        if return_class_label:
            class_label = item["annotations"][0]["category_id"]
            ret_list.append(class_label)

        return ret_list
