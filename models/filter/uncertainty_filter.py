from models.filter._base import BaseFilter
import numpy as np


class UncertaintyFilter(BaseFilter):
    # TODO implement
    def __init__(self, uncertainty_threshold=None, keep_topk_percent=None, ):
        super().__init__()
        self.is_bool_filter = False
        self.is_relative_filter = True

    def filter_function(self, mask, prob_mask, target_index):
        # if the uncertainty of the target class is low, remove
        target_probs = prob_mask[target_index]
        # when prob is max -- how confident is the model?
        average_prob = target_probs[mask != target_index].mean()
        return average_prob

    def resolve(self, score_list):
        score_list = np.array(score_list)
