import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from TADP.utils.inference import load_tadp_seg_for_inference
from datasets.ade_info import PALETTE


def visualize_prediction(pred: np.ndarray, img: np.ndarray = None):
    """If image is passed, prediction will be overlayed with opacity."""
    r = np.zeros_like(pred).astype(np.uint8)
    g = np.zeros_like(pred).astype(np.uint8)
    b = np.zeros_like(pred).astype(np.uint8)
    for i in range(len(PALETTE)):
        mask = (pred == i)
        r[mask] = PALETTE[i][0]
        g[mask] = PALETTE[i][1]
        b[mask] = PALETTE[i][2]
    color_mask = np.stack([r, g, b], axis=2)

    if img is not None:
        plt.imshow(img)
        plt.imshow(color_mask, alpha=0.5)
    else:
        plt.imshow(color_mask)
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_tadp_seg_for_inference("checkpoints/tadp_seg_blipmin40.ckpt", device=device)

    img = cv2.imread("demo/example_img.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    caption = "a large stone house with red roof and white trimming on the front of the house and a green lawn in front of the house with a fence on the side of the house and a tree in the middle of the lawn"

    pred_seg = model.single_image_inference(img, caption)

    visualize_prediction(pred_seg, img)
