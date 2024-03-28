import json
import os

import matplotlib.pyplot as plt
from PIL import Image
from models.filter._base import FilterWrapper
from models.filter.empty_filter import EmptyFilter
from datasets.segmentation.pascal import classes

data_path = 'data/synthetic_data_pascal_cartoon_v1_multiclass_GPT'
fw = FilterWrapper('data/synthetic_data_pascal_cartoon_v1_multiclass_GPT', class_list=classes, debug_mode=True)
ef = EmptyFilter(background_ratio_threshold=0.98)
fw.add_filter(ef)

fw.filter()

exclude_list = json.load(open('data/filtering/debug/exclude_list.json', 'r'))
for file in exclude_list:
    print(file)

    path_to_img = os.path.join(data_path, file + '.png')
    path_to_mask = os.path.join(data_path, file + '_mask.png')

    img = Image.open(path_to_img)
    mask = Image.open(path_to_mask)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img)
    axes[1].imshow(mask)
    plt.show()


