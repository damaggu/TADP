import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm_cross_attention.util import instantiate_from_config
from ldm_cross_attention.models.diffusion.ddim import DDIMSampler
from ldm_cross_attention.models.diffusion.plms import PLMSSampler
import torchvision
from misc.ca_analysis_utils import TextConditioningWrapper
from datasets.VOCDataset import classes
import json


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="cross_attention_analysis/"
    )
    parser.add_argument(
        "--make_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )

    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="stable_diffusion/configs/stable-diffusion/v1-cross_attention_inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stable_diffusion/models/ldm/stable-diffusion-v1-4/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    parser.add_argument(
        "--visualize_cross_attention",
        action='store_true',
        default=False
    )

    parser.add_argument(
        "--save_to_numpy",
        action='store_true',
        default=False
    )

    parser.add_argument(
        "--caption_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
    )

    parser.add_argument('--only_save_summary', action='store_true', default=False)
    parser.add_argument('--include_eos', action='store_true', default=False)
    parser.add_argument('--class_embedding_path', type=str, default='./data/pascal_class_embeddings.pth')

    #       [  1,  21,  41,  61,  81, 101, 121, 141, 161, 181, 201, 221, 241,  ## 13, 3 * 13 + 1 * 11 = 50
    #        261, 281, 301, 321, 341, 361, 381, 401, 421, 441, 461, 481, 501,
    #        521, 541, 561, 581, 601, 621, 641, 661, 681, 701, 721, 741, 761,
    #        781, 801, 821, 841, 861, 881, 901, 921, 941, 961, 981]
    # cross attention plotting parameters
    parser.add_argument('--timesteps_to_visualize', type=str, default='981')
    parser.add_argument('--include_head_average', action='store_true', default=False)
    parser.add_argument('--init_image', type=str, default=None)
    parser.add_argument('--step_range', type=str, default=None)

    # text conditioning wrapper
    parser.add_argument('--text_conditioning', type=str, default=None)
    parser.add_argument('--min_blip', type=int, default=0)
    opt = parser.parse_args()

    seed_everything(opt.seed)
    if opt.step_range is not None:
        opt.step_range = [int(t) for t in opt.step_range.split(',')]
    # process timesteps to visualize
    opt.timesteps_to_visualize = [int(t) for t in opt.timesteps_to_visualize.split(',')]

    config = OmegaConf.load(f"{opt.config}")
    config.model.params.unet_config.params.visualize_ca = opt.visualize_cross_attention
    if opt.text_conditioning == 'class_emb':
        name = '/'.join(opt.class_embedding_path.split('/')[1:]).replace('.pth', '').replace('/', '_')
        outpath = os.path.join(opt.outdir, name)
    elif opt.text_conditioning == 'class_names':
        outpath = os.path.join(opt.outdir, 'class_names')
    else:
        outpath = os.path.join(opt.outdir, opt.caption_path.split('/')[-1].replace('.json', ''))
    config.model.params.unet_config.params.visualize_ca_params = {
        'visualize_specific_timesteps': opt.timesteps_to_visualize,
        'include_head_average': opt.include_head_average,
        'output_folder': outpath,
        'only_save_summary': opt.only_save_summary,
        'save_to_numpy': opt.save_to_numpy}
    model = load_model_from_config(config, f"{opt.ckpt}")
    model.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(outpath, exist_ok=True)
    # outpath = opt.outdir

    batch_size = opt.batch_size

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = 0  # len(os.listdir(sample_path))

    def load_image(path):
        x_samples_ddim = Image.open(path)
        x_samples_ddim = torchvision.transforms.Resize((512, 512))(x_samples_ddim)
        x_samples_ddim = torch.tensor(np.array(x_samples_ddim)).permute(2, 0, 1).unsqueeze(0) / 255
        x_samples_ddim = x_samples_ddim.to('cuda')
        code = model.get_first_stage_encoding(model.encode_first_stage(x_samples_ddim))
        return code

    cfg = {'text_conditioning': opt.text_conditioning,
           'class_names': classes,
           'caption_path': opt.caption_path
           }
    tcw = TextConditioningWrapper(cfg, model, class_embedding_path=opt.class_embedding_path)

    pascal_img_path = '/home/nkondapa/PycharmProjects/neurips2023_v2/data/VOCdevkit/VOC2012/JPEGImages/'
    img_id_file = 'data/VOCdevkit/VOC2012/ImageSets/Segmentation/trainaug.txt'
    with open(img_id_file, 'r') as f:
        img_ids = f.readlines()
    img_ids = [img_id.strip() for img_id in img_ids]

    # img_ids = tcw.blip_captions.keys()
    img_ids = list(img_ids)
    # np.random.shuffle(img_ids)
    img_ids = img_ids[:opt.n_samples]
    img_ids.append('2010_001715')
    # img_names = ['base_bottle', 'base_cat', 'base_horse']
    # img_ids = img_ids[:len(img_names)]
    # img_ids = ['2010_002139',
    #            '2010_001743',
    #            '2010_001715',
    #            '2009_004645']
    # batchify
    img_id_batches = list(chunk(img_ids, batch_size))
    print(img_id_batches)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for img_id_batch in tqdm(img_id_batches, desc="data"):
                    uc = None
                    # if opt.scale != 1.0:
                    #     uc = model.get_learned_conditioning(batch_size * [""])

                    c, tokens = tcw.create_text_embeddings(img_metas={'img_id': list(img_id_batch)})
                    if opt.text_conditioning != 'class_emb' and opt.text_conditioning != 'class_names':
                        prompts = []
                        for img_id in img_id_batch:
                            prompts.append(tcw.blip_captions[img_id]['captions'])
                        bpe_decoded_prompts = [
                            [model.cond_stage_model.tokenizer.decode(tokens[j][i]) for i in range(77)] for j in
                            range(len(prompts))]
                    elif opt.text_conditioning == 'class_names':
                        bpe_decoded_prompts = [
                            [model.cond_stage_model.tokenizer.decode(tokens[j][i]) for i in range(24)] for j in
                            range(len(img_id_batch))]
                    else:
                        bpe_decoded_prompts = [classes] * batch_size

                    for key in model.model.diffusion_model.xti_cross_attention_target_index_dict:
                        key.plot_dict['sample_names'] = img_id_batch
                        # key.plot_dict['sample_names'] = img_names
                        key.plot_dict['include_eos'] = opt.include_eos # TODO check if can be moved up

                    start_codes = []
                    for ii, img_id in enumerate(img_id_batch):
                        start_codes.append(load_image(f'{pascal_img_path}/{img_id}.jpg'))
                        # start_codes.append(load_image(f'./{img_names[ii]}.jpg'))
                    start_codes = torch.cat(start_codes, dim=0)

                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=c,
                                                     batch_size=min(batch_size, len(start_codes)),
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     x_T=start_codes,
                                                     prompts=bpe_decoded_prompts,
                                                     tokens=tokens,
                                                     step_range=opt.step_range
                                                     )

                    # x_samples_ddim = model.decode_first_stage(samples_ddim)
                    # x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    if not opt.skip_save:
                        for img_id in img_id_batch:
                            img_path = os.path.join(pascal_img_path, f'{img_id}.jpg')
                            if not os.path.exists(os.path.join(sample_path, f'{img_id}.jpg')):
                                os.symlink(img_path, os.path.join(sample_path, f'{img_id}.jpg'))
                            base_count += 1

                        # for img_name in img_names:
                        #     img_path = os.path.join('./', f'{img_name}.jpg')
                        #     if not os.path.exists(os.path.join(sample_path, f'{img_name}.jpg')):
                        #         os.symlink(img_path, os.path.join(sample_path, f'{img_name}.jpg'))
                        #     base_count += 1

                        # for x_sample in x_samples_ddim:
                        #     x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        #     Image.fromarray(x_sample.astype(np.uint8)).save(
                        #         os.path.join(sample_path, f"{base_count:05}.jpg"))
                        #     base_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
