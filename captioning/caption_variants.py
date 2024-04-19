import json
import argparse
import os
import random

import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet

import torch

from datasets.VOCDataset import VOCDataset
from datasets.VOC_config import cfg as voc_cfg

from collections import Counter

from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

from tqdm import tqdm

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def get_nouns(sentence, unique_nouns_only=True):
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)
    nouns = [word for word, pos in tagged_words if pos == 'NN' or pos == 'NNS']
    if unique_nouns_only:
        nouns = list({noun: "" for noun in nouns}.keys())
    return " ".join(nouns)


def parse_args():
    parser = argparse.ArgumentParser(description='Modify a JSON file with image captions')
    parser.add_argument('--type', type=str, default='class_names')
    parser.add_argument("--remove_n_classes", type=int, default=None)
    parser.add_argument("--remove_pct_classes", type=float, default=None)
    parser.add_argument("--add_n_classes", type=int, default=None)
    parser.add_argument("--add_pct_classes", type=float, default=None)
    parser.add_argument('--source_captions', type=str, default=None)
    parser.add_argument('--caption_ext', type=str, default="")
    parser.add_argument('--dataset_name', type=str, default='pascal')
    parser.add_argument("--shuffle", action='store_true', default=False)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.76)
    args = parser.parse_args()
    return args


def filter_nouns(captions):
    new_captions = {}
    for img_id in tqdm(captions):
        img_captions = captions[img_id]['captions']
        new_img_captions = [get_nouns(caption) for caption in img_captions]
        new_captions[img_id] = {'captions': new_img_captions}
    return new_captions


def get_class_names(dataset_name, shuffle, remove_n_classes, remove_pct_classes, add_n_classes, add_pct_classes):
    if dataset_name == 'pascal':
        train_dataset = VOCDataset('../', 'VOC2012', voc_cfg, 'train', True)
        val_dataset = VOCDataset('../', 'VOC2012', voc_cfg, 'val', False)
        dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
        dataset_classes = train_dataset.classes

    new_captions = {}

    for idx in tqdm(range(len(dataset))):
        x, x_mask, img_metas = dataset[idx]
        img_id = img_metas['img_id']
        labels_list = x_mask.view(-1).to(torch.int).tolist()
        counter = Counter(labels_list)
        new_caption = []
        for label, class_name in enumerate(dataset_classes):
            if label in counter:
                new_caption.append(class_name)
                # new_caption.append(label)
        if remove_n_classes is not None:
            has_background = ('background' == new_caption[0])
            present_classes = new_caption[1:] if has_background else new_caption
            n = min(remove_n_classes, max(0, len(present_classes) - 1))
            removed_classes = random.sample(present_classes, n)
            for removed_class in removed_classes:
                new_caption.remove(removed_class)
        if remove_pct_classes is not None:
            has_background = ('background' == new_caption[0])
            present_classes = new_caption[1:] if has_background else new_caption
            removed_classes = []
            for present_class in present_classes:
                if random.random() < remove_pct_classes:
                    removed_classes.append(present_class)
            for removed_class in removed_classes:
                new_caption.remove(removed_class)
        if add_n_classes is not None:
            absent_classes = [dataset_class for dataset_class in dataset_classes if (dataset_class != 'background' and
                                                                                     dataset_class not in new_caption)]
            n = min(add_n_classes, len(absent_classes))
            added_classes = random.sample(absent_classes, n)
            for added_class in added_classes:
                new_caption.append(added_class)
            ordered_caption = [dataset_class for dataset_class in dataset_classes if dataset_class in new_caption]
            new_caption = ordered_caption
        if add_pct_classes is not None:
            absent_classes = [dataset_class for dataset_class in dataset_classes if (dataset_class != 'background' and
                                                                                     dataset_class not in new_caption)]
            added_classes = []
            for absent_class in absent_classes:
                if random.random() < add_pct_classes:
                    added_classes.append(absent_class)
            for added_class in added_classes:
                new_caption.append(added_class)
            ordered_caption = [dataset_class for dataset_class in dataset_classes if dataset_class in new_caption]
            new_caption = ordered_caption
        if shuffle:
            random.shuffle(new_caption)

        new_caption = " ".join(new_caption)
        # print(dataset_classes)
        # print(new_caption)
        new_captions[img_id] = {'captions': [new_caption]}
    return new_captions

def get_all_class_names(dataset_name):
    if dataset_name == 'pascal':
        train_dataset = VOCDataset('../', 'VOC2012', voc_cfg, 'train', True)
        val_dataset = VOCDataset('../', 'VOC2012', voc_cfg, 'val', False)
        dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
        dataset_classes = train_dataset.classes

    new_captions = {}
    new_caption = []
    for class_name in dataset_classes:
        new_caption.append(class_name)
    new_caption = " ".join(new_caption)
    for idx in tqdm(range(len(dataset))):
        _, _, img_metas = dataset[idx]
        img_id = img_metas['img_id']
        new_captions[img_id] = {'captions': [new_caption]}
    return new_captions


if __name__ == "__main__":
    args = parse_args()

    source_captions = args.source_captions
    if source_captions is not None:
        caption_filename, _ = os.path.splitext(source_captions)
    else:
        caption_filename = 'captions/' + args.dataset_name
        random.seed(args.random_seed)

    if args.type == 'filter_nouns':
        with open(source_captions, 'r') as f:
            captions = json.load(f)
        new_captions = filter_nouns(captions)
        type_ext = '_nouns_only'

    if args.type == 'class_names':
        new_captions = get_class_names(args.dataset_name, shuffle=args.shuffle,
                                       remove_n_classes=args.remove_n_classes,
                                       remove_pct_classes=args.remove_pct_classes,
                                       add_n_classes=args.add_n_classes,
                                       add_pct_classes=args.add_pct_classes)
        type_ext = '_class_names'
        if args.shuffle:
            type_ext += '_shuffle'

    if args.type == 'all_class_names':
        new_captions = get_all_class_names(args.dataset_name)
        type_ext = '_all_class_names'

    new_caption_path = caption_filename + type_ext + '.json'
    with open(new_caption_path, "w") as f:
        json.dump(new_captions, f)
