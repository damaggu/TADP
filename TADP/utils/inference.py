import numpy as np
import mmcv
from mmseg.models import build_segmentor
from mmcv.runner import wrap_fp16_model, load_checkpoint

from TADP.tadp_seg_mm import TADPSeg  # import this to register model


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


def load_tadp_seg_for_inference(ckpt_path: str):
    args = _get_seg_args()
    args.ckpt_path = ckpt_path

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
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.ckpt_path, map_location='cpu')
    return model
