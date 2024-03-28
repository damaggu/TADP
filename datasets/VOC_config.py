# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import torch
import argparse
import os
import sys
import cv2
import time


class DatasetConfiguration():
    def __init__(self):

        self.DATA_NAME = 'VOC2012'
        self.DATA_AUG = True
        self.DATA_WORKERS = 8
        self.DATA_RESCALE = 512
        self.DATA_RANDOMCROP = 512
        self.DATA_RANDOMROTATION = 0
        self.DATA_RANDOMSCALE = 2
        self.DATA_RANDOM_H = 10
        self.DATA_RANDOM_S = 10
        self.DATA_RANDOM_V = 10
        self.DATA_RANDOMFLIP = 0.5
        self.MODEL_NUM_CLASSES = 21
        self.TEST_MULTISCALE = [1.0]  # [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        self.TEST_FLIP = False  # True
        self.__check()

    def __check(self):
        if not torch.cuda.is_available():
            raise ValueError('config.py: cuda is not avalable')

    def __add_path(self, path):
        if path not in sys.path:
            sys.path.insert(0, path)


cfg = DatasetConfiguration()
