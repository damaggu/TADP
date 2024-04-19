from models.filter._base import BaseFilter
import numpy as np
import torch

class EmptyFilter(BaseFilter):

    def __init__(self, background_ratio_threshold=0.99):
        super().__init__()
        self.is_bool_filter = True
        self.background_ratio_threshold = background_ratio_threshold

    def filter_function(self, mask, prob_mask, target_index):
        # if the ratio of pixels determined to be background comprise more than x% of the image, filter returns false
        if (torch.sum((mask == 0).type(torch.long)) / float(mask.numel())) >= self.background_ratio_threshold:
            print((torch.sum((mask == 0).type(torch.long)) / float(mask.numel())))
            return False
        else:
            return True
