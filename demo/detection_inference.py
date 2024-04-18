import lightning.pytorch as pl
import torch
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from TADP.tadp_objdet import TADPObj
from TADP.utils.detection_utils import voc_classes

pl.seed_everything(42)
# load model from checkpoint

cfg = yaml.load(open("./sd_tune.yaml", "r"), Loader=yaml.FullLoader)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

cfg["freeze_text_adapter"] = True
cfg['blip_caption_path'] = 'captions/voc_extended_train_val_captions.json'
cfg['cross_blip_caption_path'] = 'captions/watercolor_captions.json'
cfg["append_self_attention"] = False
cfg['text_conditioning'] = "blip"
cfg['textual_inversion_token_path'] = "model_personalization/tokens/water_color_50/"
cfg['textual_inversion_caption_path'] = "textual_inversion_captions/voc_extended_captions.json"
cfg['dreambooth_checkpoint'] = None
cfg['val_dataset_name'] = "watercolor"
cfg['use_scaled_encode'] = True

load_checkpoint_path = '/mnt/m4_data/watercolor_checkpoints/checkpoint_epoch_2.pt'

detection_model = TADPObj(class_embedding_path="./data/pascal_class_embeddings.pth", cfg=cfg, class_names=voc_classes)
detection_model.to(device)
detection_model.load_state_dict(torch.load(load_checkpoint_path), strict=False)
detection_model.eval()

img = Image.open("demo/189366231.jpg")

# #### testing model
# x = torch.randn(1, 3, 512, 512).to(device)
a = detection_model.inference([img], captions=["a car on the road"])

boxes = a[0]['boxes'].detach().cpu().numpy()
labels = a[0]['labels'].detach().cpu().numpy()
scores = a[0]['scores'].detach().cpu().numpy()
# threshold boxes at 0.5
threshold = 0.5
boxes = boxes[scores > threshold]
labels = labels[scores > threshold]
scores = scores[scores > threshold]

# resize image to 512x512
img = img.resize((512, 512))

fig, ax = plt.subplots(1)
ax.imshow(img)
for box, label, score in zip(boxes, labels, scores):
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                             facecolor='none')
    ax.add_patch(rect)
    ax.text(box[0], box[1], f'{label} {score:.2f}', color='r', fontsize=8)
plt.show()

print(a)
