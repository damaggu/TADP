from PIL import Image
import matplotlib.pyplot as plt

from TADP.utils.inference import load_tadp_for_depth_inference

model = load_tadp_for_depth_inference("checkpoints/tadp_depth_blipmin40.ckpt")

img = Image.open("demo/example_img.jpg")
caption = "a large stone house with red roof and white trimming on the front of the house and a green lawn in front of the house with a fence on the side of the house and a tree in the middle of the lawn"

pred_depth = model.single_image_inference(img, caption)

plt.imshow(pred_depth, cmap="magma")
plt.show()
