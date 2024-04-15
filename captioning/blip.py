import os
import json
import time

import torch
from tqdm import tqdm
from pytorch_lightning import seed_everything
from transformers import AutoProcessor, Blip2ForConditionalGeneration


class BLIPInterface:
    def __init__(self, caption_dataset, caption_dataset_name, caption_params_name, blip_generation_param_dict=None, synthetic_data_folder=None):

        self.model = None
        self.processor = None
        self.device = None
        if blip_generation_param_dict is None:
            self.blip_generation_param_dict = {
                'max_new_tokens': 20,
                'min_new_tokens': 0,
            }
        else:
            self.blip_generation_param_dict = blip_generation_param_dict
        self.caption_dataset = caption_dataset
        if not hasattr(self.caption_dataset, 'get_image'):
            raise ValueError('caption_dataset must have get_image() method. '
                             'This method should return PIL images and should not apply transforms.')
        self.caption_dataset_name = caption_dataset_name
        self.caption_params_name = caption_params_name
        self.synthetic_data_folder = synthetic_data_folder
        self.out_path= "../captions/"
        if self.caption_params_name is not None:
            self.captions_file = os.path.join(self.out_path, self.caption_dataset_name + '_captions_' + self.caption_params_name + '.json')
        else:
            self.captions_file = os.path.join(self.out_path, self.caption_dataset_name + '_captions.json')

    def init_blip_model(self):
        # Takes a while to load
        st = time.time()
        print('Loading Blip2Processor...')
        processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        print('Loading Blip2ForConditionalGeneration... (can take a minute)')
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        print(f'Loaded Blip2 in {time.time() - st} seconds')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        self.model = model
        self.processor = processor
        self.device = device

    def load_blip_captions(self, overwrite=False):
        if overwrite:
            return None
        try:
            folder_name = self.out_path
            os.makedirs(folder_name, exist_ok=True)
            with open(self.captions_file, 'r') as f:
                image_file_captions = json.load(f)
                print(f'Loaded {len(image_file_captions)} blip captions from {self.caption_dataset_name} dataset')
                return image_file_captions
        except FileNotFoundError:
            return None

    def __call__(self, batch_size=128, repeat_image=1, seed=42, overwrite=False, img_name_dict=None, profiling=False):

        image_file_captions = self.load_blip_captions(overwrite=overwrite)
        if image_file_captions is not None:
            return image_file_captions

        self.init_blip_model()
        seed_everything(seed)
        if repeat_image != 1:
            raise NotImplementedError

        image_file_captions = {}
        num_images = len(self.caption_dataset)
        with torch.no_grad():
            for i in tqdm(range(0, num_images, batch_size)):
                image_batch_indices = list(range(i, min(i + batch_size, num_images)))
                image_batch = [self.caption_dataset.get_image(j, return_img_file=True) for j in image_batch_indices]
                image_batch, image_files = zip(*image_batch)
                inputs = self.processor(image_batch, return_tensors="pt").to(self.device, torch.float16)
                generated_ids = self.model.generate(**inputs,
                                                    max_new_tokens=self.blip_generation_param_dict['max_new_tokens'],
                                                    min_new_tokens=self.blip_generation_param_dict['min_new_tokens'])
                generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

                for j, (img_file, caption) in enumerate(zip(image_files, generated_texts)):
                    # only image name
                    if img_name_dict is None:
                        img_name = img_file.split('/')[-1].split('.')[0]
                    else:
                        img_name = img_name_dict[img_file]

                    if img_name not in image_file_captions:
                        # fix uint8 to int
                        image_file_captions[img_name] = {'captions': []}
                    # Remove special tokens at the end
                    caption = caption[:-1]
                    image_file_captions[img_name]['captions'].append(caption)

        if not profiling:
            os.makedirs(self.out_path, exist_ok=True)
            with open(self.captions_file, 'w') as f:
                # json.dump(image_file_captions, f, indent=2)
                json.dump(image_file_captions, f)

        return image_file_captions