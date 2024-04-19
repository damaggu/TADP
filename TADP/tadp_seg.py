import os
import json
from itertools import islice

import torch
import torch.nn as nn

from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from segmentation_models_pytorch.base import SegmentationHead

import numpy as np

import torchvision.transforms as T
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
from segmentation_models_pytorch.decoders.fpn.decoder import FPNBlock, SegmentationBlock, MergeBlock
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder as original_FPNDecoder

from models.segmentation._abstract import SegmentationModel
from ldm.util import instantiate_from_config
from TADP.vpd.models import UNetWrapper, TextAdapter


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


#### adapted from
# https://github.com/wl-zhao/VPD
class FPNDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            encoder_depth=5,
            pyramid_channels=256,
            segmentation_channels=128,
            dropout=0.2,
            merge_policy="add",
    ):
        super().__init__()

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        # encoder_channels = encoder_channels[::-1]
        self.encoder_channels = encoder_channels[: encoder_depth + 1]

        # do automaticallyy instead
        for i in range(1, len(self.encoder_channels)):
            setattr(self, f"p{i + 1}", FPNBlock(pyramid_channels, self.encoder_channels[i - 1]))
        setattr(self, f"p{len(self.encoder_channels) + 1}",
                nn.Conv2d(self.encoder_channels[-1], pyramid_channels, kernel_size=1))

        self.seg_blocks = nn.ModuleList(
            [
                SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
                for n_upsamples in [3, 2, 1, 0]
            ]
        )

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, *features):

        # replace throghu loop
        ps = []
        k = -1
        for i in reversed(range(2, len(features) + 2)):
            if i == len(features) + 1:
                p = getattr(self, f"p{i}")(features[k])
            else:
                p = getattr(self, f"p{i}")(p, features[k])
            k -= 1
            ps.append(p)

        # feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, ps)]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)

        return x


class TADPSeg(SegmentationModel):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 decode_head=None,
                 sd_path='checkpoints/v1-5-pruned-emaonly.ckpt',
                 unet_config=dict(),
                 class_embedding_path='ade_class_embeddings.pth',
                 gamma_init_value=1e-4,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 use_decoder=False,
                 cfg=None,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        # get config from *args and **kwargs
        self.cfg = cfg
        self.cfg['original_tc_str'] = self.cfg['text_conditioning']
        self.texts = []

        # turn text conditioning into list
        if '+' in self.cfg['text_conditioning']:
            self.cfg['text_conditioning'] = self.cfg['text_conditioning'].split('+')
        else:
            self.cfg['text_conditioning'] = [self.cfg['text_conditioning']]

        ### check if model is there if not DL
        self.text2imgModel = None
        ckpt = "v1-5-pruned-emaonly.ckpt"
        repo = "runwayml/stable-diffusion-v1-5"
        out_dir = "checkpoints"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(os.path.join(out_dir, ckpt)):
            hf_hub_download(repo_id="runwayml/stable-diffusion-v1-5", filename=ckpt, local_dir=out_dir)

        config = OmegaConf.load('stable_diffusion/configs/stable-diffusion/v1-inference.yaml')
        config.model.params.ckpt_path = f'./{sd_path}'

        # For present_class_embeds_only
        self.present_class_embeds_only = cfg['present_class_embeds_only']

        if self.present_class_embeds_only:
            temp_model = instantiate_from_config(config.model).to('cuda')
            empty_str_clip_output = temp_model.get_learned_conditioning([""])
            self.empty_class_embed = empty_str_clip_output[0, -1]

        if self.cfg['original_tc_str'] == 'class_embs':
            config.model.params.cond_stage_config.target = 'stable_diffusion.ldm.modules.encoders.modules.AbstractEncoder'

        # prepare the unet
        # sys.path.append('./stable_diffusion/ldm')
        # sd_model = instantiate_from_config(config.model)
        sd_model = instantiate_from_config(config.model)

        # handle logic for using scaled encoder
        if not self.cfg.get('use_scaled_encode', False):
            self.encoder_vq = sd_model.first_stage_model
            sd_model.first_stage_model = None
            if self.cfg['use_decoder_features']:
                use_decoder = True
            if not use_decoder:
                del self.encoder_vq.decoder
            ### set grads to zero to be safe
            for param in self.encoder_vq.parameters():
                param.requires_grad = False
        else:
            if not use_decoder:
                del sd_model.first_stage_model.decoder

        self.model = UNetWrapper(sd_model.model, **unet_config)
        sd_model.model = None
        keep_cond = False
        if 'blip' in self.cfg["original_tc_str"]:
            with open(self.cfg['blip_caption_path'], 'r') as f:
                self.blip_captions = json.load(f)
                # get max length
                self.blip_max_length = max([len(caption) for caption in self.blip_captions])
            keep_cond = True

        if self.cfg['present_class_embeds_only']:
            with open(self.cfg['present_classes_path'], 'r') as f:
                self.present_classes = json.load(f)

        if 'class_names' in self.cfg['text_conditioning']:
            with torch.no_grad():
                sd_model.cond_stage_model.to('cuda')
                class_emb_stack = []
                all_pos = 0
                eos_token = 49407
                for i, class_name in enumerate(self.class_names):
                    _emb, tokens = sd_model.get_learned_conditioning(class_name, return_tokens=True)
                    if len(class_emb_stack) == 0:
                        eos_pos = torch.where(tokens == eos_token)[1][0].item()
                        all_pos = eos_pos
                        class_emb_stack.append(_emb[:, :eos_pos])
                    else:
                        eos_pos = torch.where(tokens == eos_token)[1][0].item()
                        all_pos += (eos_pos - 1)
                        class_emb_stack.append(_emb[:, 1:eos_pos])

                self.class_names_embs = torch.cat(class_emb_stack, dim=1)

        if not keep_cond:
            del sd_model.cond_stage_model
        else:
            if self.cfg['cond_stage_trainable']:
                for param in sd_model.cond_stage_model.parameters():
                    param.requires_grad = True
            else:
                for param in sd_model.cond_stage_model.parameters():
                    param.requires_grad = False

        self.use_decoder = use_decoder
        self.sd_model = sd_model

        # check if class_embedding_path exists
        if not os.path.exists(class_embedding_path):
            print('No class embeddings provided!, please run create_class_embeddings.py --dataset pascal')

        class_embeddings = torch.load(class_embedding_path)
        self.register_buffer('class_embeddings', class_embeddings)
        text_dim = class_embeddings.size(-1)
        self.gamma = nn.Parameter(torch.ones(text_dim) * gamma_init_value)
        self.text_adapter = TextAdapter(text_dim=text_dim)

        # check if using the correct class embeddings
        assert class_embeddings.size(0) == self.n_classes

        self.with_neck = True
        self.decode_head = nn.Module()
        enc_mid_channels, enc_end_channels = self.compute_decoder_head_shapes()

        if self.cfg["decode_head"] == 'FPN':
            if self.cfg['use_decoder_features']:
                self.decode_head.decoder = FPNDecoder(
                    encoder_channels=(384, 512, 512, 1856, enc_mid_channels, enc_end_channels, 1280),
                    encoder_depth=7,
                    pyramid_channels=256,
                    segmentation_channels=128,
                    dropout=0.2,
                    merge_policy="add",
                )
            else:
                self.decode_head.decoder = original_FPNDecoder(
                    encoder_channels=(320, enc_mid_channels, enc_end_channels, 1280),
                    encoder_depth=4,
                    pyramid_channels=256,
                    segmentation_channels=128,
                    dropout=0.2,
                    merge_policy="add",
                )
        elif self.cfg["decode_head"] == 'deeplabv3plus':
            self.decoder = DeepLabV3PlusDecoder(
                encoder_channels=(320, enc_mid_channels, enc_end_channels, 1280),
                out_channels=256,
                atrous_rates=(12, 24, 36),
                output_stride=16,
            )
        else:
            raise NotImplementedError

        self.decode_head.segmentation_head = SegmentationHead(
            in_channels=self.decode_head.decoder.out_channels,
            out_channels=self.n_classes,
            activation=None,
            kernel_size=1,
            upsampling=8,
        )
        self.decode_head.num_classes = self.n_classes

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if self.cfg["freeze_text_adapter"]:
            for param in self.text_adapter.parameters():
                param.requires_grad = False

        if self.cfg["use_token_embeds"]:
            self.reduce_embeds = nn.Linear(768, 8)

    def initialize_model(self):
        pass

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def compute_decoder_head_shapes(self):
        text_cond = self.cfg['text_conditioning']
        enc_mid_channels = 640
        enc_end_channels = 1280
        if self.cfg['append_self_attention']:
            enc_mid_channels += 1024
            enc_end_channels += 256

        if self.cfg.get('use_attn', True):
            if 'blip' in text_cond:
                enc_mid_channels += 77
                enc_end_channels += 77

            if 'class_names' in text_cond:
                enc_mid_channels += self.class_names_embs.shape[1]
                enc_end_channels += self.class_names_embs.shape[1]

            if 'class_emb' in text_cond:
                enc_mid_channels += self.n_classes
                enc_end_channels += self.n_classes

        return enc_mid_channels, enc_end_channels

    def create_text_embeddings(self, img_metas, latents):
        bsz = len(latents)
        text_cond = self.cfg['text_conditioning']
        conds = []
        texts = None
        if 'blip' in text_cond:
            texts = []
            _cs = []
            for img_id in img_metas['img_id']:
                text = self.blip_captions[img_id]['captions']
                c = self.sd_model.get_learned_conditioning(text)
                texts.append(text)
                _cs.append(c)
            c = torch.cat(_cs, dim=0)
            conds.append(c)

        if 'blank_str' in text_cond:
            texts = []
            _cs = []
            for img_id in img_metas['img_id']:
                text = ['']
                c = self.sd_model.get_learned_conditioning(text)
                texts.append(text)
                _cs.append(c)
            c = torch.cat(_cs, dim=0)
            conds.append(c)

        if 'class_names' in text_cond:
            _cs = []
            for img_id in img_metas['img_id']:
                _cs.append(self.class_names_embs)
            c = torch.cat(_cs, dim=0)
            conds.append(c)

        if 'class_emb' in text_cond:
            c_crossattn = self.class_embeddings.repeat(bsz, 1, 1).clone()

            if self.present_class_embeds_only:
                for img_idx, img_id in enumerate(img_metas['img_id']):
                    present_classes = self.present_classes[img_id]['captions'][0]
                    # print(img_idx, img_id, present_classes)
                    for class_idx in range(c_crossattn.shape[1]):
                        if class_idx not in present_classes:
                            c_crossattn[img_idx, class_idx, :] = self.empty_class_embed
            conds.append(c_crossattn)

        c_crossattn = torch.cat(conds, dim=1)
        if self.cfg['use_text_adapter']:
            c_crossattn = self.text_adapter(c_crossattn, self.gamma)

        if texts is not None:
            self.texts = texts

        return c_crossattn

    def extract_feat(self, img, img_metas):
        """Extract features from images."""
        if self.cfg.get('use_scaled_encode', False):
            with torch.no_grad():
                latents = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(img))
        else:
            with torch.no_grad():
                latents = self.encoder_vq.encode(img)
            latents = latents.mode().detach()

        c_crossattn = self.create_text_embeddings(img_metas, latents)

        t = torch.from_numpy(np.array([1])).to(img.device)
        outs = self.model(latents, t, c_crossattn=[c_crossattn])

        if self.cfg['use_decoder_features']:
            decoded, decoder_outs = self.encoder_vq.decode_blocks(latents)

            # stack decoder outs[:3] along first dim
            chan_64 = torch.cat(decoder_outs[:3], dim=1)
            outs[0] = torch.cat([outs[0], chan_64], dim=1)
            outs.insert(0, decoder_outs[3])
            outs.insert(0, decoder_outs[4])
            outs.insert(0, torch.cat(decoder_outs[5:], dim=1))

        return outs

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img, img_metas)

        if self.with_neck:
            x = self.neck(x)

        return x

    def forward(self, x, img_metas):
        if self.normalize_images:
            # x = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
            x = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(x)
        x = self.extract_feat(x, img_metas)

        if self.with_neck:
            # TODO: needs to be adjusted for deeplabv3+ architecture
            x = self.decode_head.decoder(*x)
            x = self.decode_head.segmentation_head(x)

        return x

    def initialize_loss(self):
        loss = smp.losses.FocalLoss(mode="multiclass", ignore_index=self.ignore_index)
        return loss

    def configure_optimizers(self):
        lesslr_no_decay = list()
        lesslr_decay = list()
        no_lr = list()
        no_decay = list()
        decay = list()
        for name, m in self.named_parameters():
            if 'unet' in name and 'norm' in name:
                lesslr_no_decay.append(m)
            elif 'unet' in name:
                lesslr_decay.append(m)
            elif 'encoder_vq' in name:
                no_lr.append(m)
            elif 'norm' in name:
                no_decay.append(m)
            elif 'embedding_manager' in name:
                pass
            else:
                decay.append(m)

        params_to_optimize = [
            {'params': lesslr_no_decay, 'weight_decay': 0.0, 'lr_scale': 0.01},
            {'params': lesslr_decay, 'lr_scale': 0.01},
            {'params': no_lr, 'lr_scale': 0.0},
            {'params': no_decay, 'weight_decay': 0.0},
            {'params': decay},
        ]
        optimizer = torch.optim.AdamW(params_to_optimize,
                                      lr=0.00001,
                                      weight_decay=1e-2,
                                      amsgrad=False
                                      )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: (1 - x / (
                self.cfg["dataset_len"] * self.cfg["max_epochs"])) ** 0.9)

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch", "frequency": 1}]

    def training_step(self, batch, batch_idx):
        images, masks, img_metas = batch
        loss = self._step(images, masks, img_metas, "train")
        sch = self.lr_schedulers()
        sch.step()

        return loss






