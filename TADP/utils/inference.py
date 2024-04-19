from collections import OrderedDict

import numpy as np
import torch
import mmcv
from mmseg.models import build_segmentor
from mmcv.runner import wrap_fp16_model, load_checkpoint

from TADP.tadp_seg_mm import TADPSeg  # import this to register model
from TADP.tadp_depth import TADPDepth
from models.depth.configs.test_options import TestOptions


class ArgNamespace():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _get_seg_args():
    return ArgNamespace(
        config="TADP/mm_configs/seg_ade20k_full.py",
        ckpt_path=None,
        text_conditioning="prompt_input",
        use_scaled_encode=True,
        use_text_adapter=False,
        debug=False,
        textual_inversion_token_path=None,
        textual_inversion_caption_path=None,
        blip_caption_path=None,
        cross_blip_caption_path=None,
        append_self_attention=False,
        work_dir=None,
        aug_test=False,
        out=None,
        formal_only=False,
        eval="mIoU",
        show=False,
        gpu_collect=False,
        gpu_id=0,
        tmpdir=None,
        options=None,
        cfg_options=None,
        eval_options=None,
        launcher="none",
        opacity=0.5,
        local_rank=0
    )


def load_tadp_seg_for_inference(ckpt_path: str, device="cuda", additional_args=None):
    args = _get_seg_args()
    args.ckpt_path = ckpt_path
    if additional_args is not None:
        args.__dict__.update(additional_args)

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.model['opt_dict'] = {
        'use_scaled_encode': args.use_scaled_encode,
        'append_self_attention': args.append_self_attention,
        'use_text_adapter': args.use_text_adapter,
        'text_conditioning': args.text_conditioning,
        'blip_caption_path': args.blip_caption_path,
        'textual_inversion_token_path': args.textual_inversion_token_path,
        'textual_inversion_caption_path': args.textual_inversion_caption_path,
        'cross_blip_caption_path': args.cross_blip_caption_path,
        'dreambooth_checkpoint': None,
    }

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.eval()
    model.to(device)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.ckpt_path, map_location='cpu')
    return model


def _get_depth_args():
    opt = TestOptions()
    args = opt.initialize().parse_args()
    args.rank = 0
    args.batch_size = 1
    args.max_depth = 10.0
    args.max_depth_eval = 10.0
    args.weight_decay = 0.1
    args.num_filters = [32, 32, 32]
    args.deconv_kernels = [2, 2, 2]
    args.save_model = False
    args.layer_decay = 0.9
    args.drop_path_rate = 0.3
    # args.log_dir = None
    args.crop_h = 480
    args.crop_w = 480
    args.epochs = 25
    args.shift_window_test = True
    args.shift_size = 2
    args.flip_test = True
    args.use_scaled_encode = True
    args.text_conditioning = "prompt_input"
    args.use_text_adapter = False
    args.trim_edges = True
    return args


def load_tadp_for_depth_inference(ckpt_path: str, device="cuda"):
    args = _get_depth_args()
    args.ckpt_dir = ckpt_path

    model = TADPDepth(args=args)
    model_weight = torch.load(args.ckpt_dir)['model']
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight, strict=False)
    model.to(device)
    model.eval()
    return model
