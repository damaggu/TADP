# # path to full_data ./data/cross-domain-detection/datasets/watercolor/JPEGImages
full_data_path = './data/cross-domain-detection/datasets/watercolor/JPEGImages'

# path to 5 data reambooth_data/watercolor_5
subset_data_path = './dreambooth_data/watercolor'

# get 5 random images from full_data and copy them to 5_data
import os
files = os.listdir(full_data_path)

# fix random seed for reproducibility


import random

random.seed(42)
random.shuffle(files)

selected_files = files[:50]

subsets = [5,10, 20, 50]

for subset in subsets:
    selected_files = files[:subset]
    for file in selected_files:
        src = os.path.join(full_data_path, file)
        dst_path = subset_data_path + "_" + str(subset)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        dst = os.path.join(dst_path, file)
        import shutil
        shutil.copyfile(src, dst)
        print(f'copied {src} to {dst}')
