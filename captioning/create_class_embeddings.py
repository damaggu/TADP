import argparse
import copy
import os
import os.path as osp
import time
import torch
import torchvision
# import denseclip
from tqdm import tqdm

from glob import glob
import json

from datasets.VOCDataset import classes

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pascal', choices=['pascal', 'coco', 'cityscapes'])

    args = parser.parse_args()
    dataset = args.dataset
    import sys
    sys.path.append('../../')
    from diff_misc import FrozenCLIPEmbedder

    # class_name = [path.split('/')[-1] for path in paths]

    if dataset == 'pascal':
        class_name = classes
    elif dataset == 'cityscapes':
        class_name = torchvision.datasets.Cityscapes.classes
        names_and_ids = tuple(zip([c.name for c in class_name], [c.train_id for c in class_name]))
        names_and_ids = sorted(names_and_ids, key=lambda x: x[1])
        class_name = [n for n, i in names_and_ids if i not in [-1, 255]]
    elif dataset == 'coco':
        class_name = [path.split('/')[-1] for path in glob('./data/coco/train2017/*')]
    else:
        raise NotImplementedError
    print(class_name)


    class_names_out = './data/{}_class_names.json'.format(dataset)
    if not osp.exists(osp.dirname(class_names_out)):
        os.makedirs(osp.dirname(class_names_out))

    with open(class_names_out, 'w') as f:
        f.write(json.dumps(class_name))

    imagenet_classes = [name.replace('_', ' ') for name in class_name]

    print(f"{len(imagenet_classes)} classes, {len(imagenet_templates)} templates")

    text_encoder = FrozenCLIPEmbedder(max_length=20)
    text_encoder.cuda()

    classnames = imagenet_classes

    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = []
            texts = texts + [template.format(classname) for template in imagenet_templates]  # format with class
            print(texts[0])
            class_embeddings = text_encoder.encode(texts).detach()
            # class_embeddings = class_embeddings.mean(dim=0)
            zeroshot_weights.append(class_embeddings)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0)

    print(zeroshot_weights.shape)
    torch.save(zeroshot_weights.mean(1).cpu(), './data/{}_class_embeddings.pth'.format(dataset))

    save_specific_prompts = []
    for pi in save_specific_prompts:
        prompt = imagenet_templates[pi]
        print(prompt)
        os.makedirs('./data/prompt={}'.format(pi), exist_ok=True)
        torch.save(zeroshot_weights[:, pi].cpu(), './data/prompt={}/{}_class_embeddings.pth'.format(pi, dataset))


if __name__ == '__main__':
    main()
