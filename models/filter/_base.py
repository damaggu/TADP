import os
from typing import Union, List
from PIL import Image
import torchvision
import torch
import json
import numpy as np

class FilterWrapper:

    def __init__(self, data_dir, class_list, filtering_save_name='debug', save_dir='./data/filtering/', debug_mode=False):
        self.data_dir = data_dir
        self.folders = os.listdir(data_dir)
        self.class_list = class_list
        self.debug_mode = debug_mode
        if filtering_save_name == 'debug' and not self.debug_mode:
            raise ValueError('filtering save name must be passed unless in debug mode')

        self.save_dir = save_dir
        self.filtering_save_name = filtering_save_name
        self.save_path = os.path.join(self.save_dir, self.filtering_save_name)
        if os.path.isdir(self.save_path) and not self.debug_mode:
            raise FileExistsError('This folder already exists!')
        os.makedirs(self.save_path, exist_ok=self.debug_mode)

        self.images, self.prob_masks, self.masks, self.classes, self.file_names = self.load_data()
        self.filter_function_list = []

    def extract_file_names(self, files):

        file_names = []
        for file in files:
            if '.npy' in file:
                continue
            if 'mask' in file:
                continue
            if '.txt' in file:
                continue
            file_names.append(file.replace('.png', ''))
        file_names = sorted(file_names)
        return file_names

    def load_data(self):

        images = []
        masks = []
        prob_masks = []
        classes = []
        file_names_all = []

        for folder in self.folders:
            print(f'Loading {folder} folder...')
            class_folder = os.path.join(self.data_dir, folder)
            files = os.listdir(class_folder)
            file_names = self.extract_file_names(files)
            try:
                num_files = 0
                for file_name in file_names:

                    if self.debug_mode:
                        if num_files == 10:
                            break

                    file_names_all.append(os.path.join(folder, file_name))
                    num_files += 1

                    mask_path = os.path.join(self.data_dir, folder, file_name + '_mask.png')
                    mask = Image.open(mask_path).convert("L")
                    mask = torchvision.transforms.functional.to_tensor(mask)

                    masks.append(mask)

                    img_name = os.path.join(self.data_dir, folder, file_name + '.png')
                    image = Image.open(img_name).convert("RGB")
                    images.append(image)

                    cls = folder
                    cls_index = self.class_list.index(cls)
                    classes.append(cls_index)

                    prob_mask_path = os.path.join(self.data_dir, folder, file_name + '_prob_mask.npy')
                    prob_mask = np.load(prob_mask_path)
                    prob_mask = torch.from_numpy(prob_mask)
                    prob_masks.append(prob_mask)

            except Exception as e:
                print(e)
                continue

        return images, prob_masks, masks, classes, file_names_all

    def filter(self):
        keep_bools = []
        filter_func_dict = {}

        for mi, masks in enumerate(self.masks):
            mask = self.masks[mi]
            prob_mask = self.prob_masks[mi]
            class_folder = self.file_names[mi].split('/')[0]

            keep = True
            for filter_func in self.filter_function_list:
                if filter_func.is_bool_filter:
                    if not filter_func(mask, prob_mask, self.class_list.index(class_folder)):
                        keep = False
                        break
                elif filter_func.is_relative_filter:
                    if filter_func not in filter_func_dict:
                        filter_func_dict[filter_func] = []
                    filter_func_dict[filter_func].append(filter_func(mask, prob_mask, self.class_list.index(class_folder)))

            if keep:
                keep_bools.append(True)
            else:
                keep_bools.append(False)

        for mi, masks in enumerate(self.masks):
            if keep_bools[mi]:
                for filter_func in filter_func_dict:
                    if not filter_func.resolve(filter_func_dict[filter_func]):
                        keep_bools[mi] = False

        self.save(keep_bools)

    def save(self, keep_bools):

        keep_list = []
        exclude_list = []
        for fi, file_name in enumerate(self.file_names):
            if keep_bools[fi]:
                keep_list.append(file_name)
            else:
                exclude_list.append(file_name)

        with open(os.path.join(self.save_path, 'keep_list.json'), 'w') as f:
            json.dump(keep_list, f, indent=2)
        with open(os.path.join(self.save_path, 'exclude_list.json'), 'w') as f:
            json.dump(exclude_list, f, indent=2)

    def add_filter(self, filter_func):
        self.filter_function_list.append(filter_func)


class BaseFilter:

    def __init__(self):
        self.is_bool_filter = False
        self.is_relative_filter = False
        pass

    def filter_function(self, mask: torch.LongTensor, prob_mask: torch.FloatTensor, target_index: int) -> Union[bool, float]:
        raise NotImplementedError

    def resolve(self, score_list: List[float]):
        raise NotImplementedError

    def __call__(self, *args):
        return self.filter_function(*args)