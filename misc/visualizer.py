from typing import Dict

import numpy as np
from scipy.interpolate import NearestNDInterpolator
import torch
from misc.color_palettes import get_palette


def make_axes_invisible(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

def save_tensor_mapping(x: torch.Tensor, m: Dict, void_val=-99):
    assert void_val not in m.values()
    assert void_val not in m.keys()
    assert (x == void_val).sum() == 0
    out = torch.full_like(x, void_val)
    for k, v in m.items():
        out[x == k] = v
    assert torch.isnan(out).sum() == 0
    return out


class SegmentationMapVisualizer:
    """Better than mmpose crap"""

    def __init__(self, cls_map=None, fill_val="black", palette="voc"):

        self._palette = get_palette(palette)

        assert fill_val in ["black", "white", "inpaint"]
        self.fill_val = fill_val
        self.cls_map = cls_map

    def palette(self, idx):
        if idx == 255 or idx == -1:
            if self.fill_val == "black":
                return 0, 0, 0
            elif self.fill_val == "white":
                return 255, 255, 255
        return self._palette[idx]

    def __call__(self, mask):

        if self.cls_map is not None:
            mask = save_tensor_mapping(mask, self.cls_map)

        if self.fill_val == "inpaint":
            inpainted_masks = []
            for m in mask:
                m = m.cpu().numpy()
                fillmask = np.where((m != 255) & (m != -1))
                interp = NearestNDInterpolator(np.transpose(fillmask), m[fillmask])
                filled_data = interp(*np.indices(m.shape))
                inpainted_masks.append(filled_data)
            mask = torch.from_numpy(np.stack(inpainted_masks))

        r = save_tensor_mapping(mask, {x: self.palette(x)[0] for x in mask.unique()}).unsqueeze(1)
        g = save_tensor_mapping(mask, {x: self.palette(x)[1] for x in mask.unique()}).unsqueeze(1)
        b = save_tensor_mapping(mask, {x: self.palette(x)[2] for x in mask.unique()}).unsqueeze(1)

        return torch.cat([r, g, b], dim=1)