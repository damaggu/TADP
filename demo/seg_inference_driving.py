import cv2
import torch

from TADP.utils.inference import load_tadp_seg_for_inference
from demo.seg_inference import visualize_prediction

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
additional_args = {
    "textual_inversion_token_path": "tokens/dark_zurich_full_nightToken_style/",
    "config": "TADP/mm_configs/fpn_vpd_sd1-5_citscapes_to_nighttime_512x512_gpu8x2_80k.py",
}
model = load_tadp_seg_for_inference("checkpoints/tadp_nighttimeDriving.pth", device=device,
                                    additional_args=additional_args)

img = cv2.imread("demo/0_frame_0205_leftImg8bit.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
caption = "a person crossing the street, a car driving on the road, a traffic light, a train in the background"

pred_seg = model.single_image_inference(img, caption)

visualize_prediction(pred_seg, img)
