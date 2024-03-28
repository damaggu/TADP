import gc
import os
import pickle
import wandb
import torch
from omegaconf import OmegaConf

import torch.nn as nn
import lightning.pytorch as pl

### tadp crossdomain object detection
from huggingface_hub import hf_hub_download

from detectron2.structures.instances import Instances
from pascal_voc_evaluation import PascalVOCDetectionEvaluator
from faster_rcn_tests import CustomFasterRCNN
from VPD.vpd import UNetWrapper, TextAdapter

from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder

import segmentation_models_pytorch as smp

from segmentation_models_pytorch.base import SegmentationHead

from ldm.util import instantiate_from_config

from itertools import islice
from PIL import Image
import numpy as np
import json

# our stuff
from diffusers.utils import _get_model_file, DIFFUSERS_CACHE

from TADP.utils import create_bounding_boxes_from_masks, annotations_to_boxes

import torch.nn.functional as F
from chainercv.evaluations import eval_detection_voc
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import torchvision.transforms as T
import torchvision.transforms.v2.functional
from matplotlib import pyplot as plt, patches
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
from torchvision import datapoints
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork

from models.segmentation._abstract import SegmentationModel

from segmentation_models_pytorch.decoders.fpn.decoder import FPNBlock, SegmentationBlock, MergeBlock
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder as original_FPNDecoder

from TADP.utils import REDUCED_CLASS_NAMES, CLASS_NAMES, DETECTRON_VOC_CLASS_NAMES


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
                 class_embedding_path='class_embeddings.pth',
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


hacky = False


class TADPObj(pl.LightningModule):

    def __init__(self,
                 decode_head=None,
                 sd_path='checkpoints/v1-5-pruned-emaonly.ckpt',
                 unet_config=dict(),
                 class_embedding_path='class_embeddings.pth',
                 gamma_init_value=1e-4,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 use_decoder=False,
                 cfg=None,
                 class_names=None,
                 freeze_backbone=False,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        # get config from *args and **kwargs

        mAP = MeanAveragePrecision()

        self.all_predictions = []
        self.all_targets = []

        self.freeze_backbone = freeze_backbone

        self.n_classes = len(class_names)
        self.metric = mAP
        self.dataset_name = "voc"

        self.cfg = cfg
        try:
            self.object_dataloader_indices = kwargs['object_dataloader_indices']
        except KeyError:
            self.object_dataloader_indices = None

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
        if ('blip' not in self.cfg["text_conditioning"] and
                'class_names' not in self.cfg['text_conditioning'] and
                'textual_inversion' not in self.cfg['text_conditioning']):
            config.model.params.cond_stage_config.target = 'stable_diffusion.ldm.modules.encoders.modules.AbstractEncoder'

        sd_model = instantiate_from_config(config.model)

        if self.cfg['dreambooth_checkpoint'] is not None:
            sd_model.load_state_dict(torch.load(self.cfg['dreambooth_checkpoint'])['state_dict'], strict=False)
            print('Loaded dreambooth checkpoint!')

        # handle logic for using scaled encoder
        if not self.cfg.get('use_scaled_encode', False):
            self.encoder_vq = sd_model.first_stage_model
            sd_model.first_stage_model = None
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
        if 'blip' in self.cfg["text_conditioning"]:
            with open(self.cfg['blip_caption_path'], 'r') as f:
                self.blip_captions = json.load(f)
                # get max length
                self.blip_max_length = max([len(caption) for caption in self.blip_captions])
            for param in sd_model.cond_stage_model.parameters():
                param.requires_grad = False
            keep_cond = True
        # if 'textual_inversion' in self.cfg["text_conditioning"]:
        #     with open(self.cfg['textual_inversion_caption_path'], 'r') as f:
        #         print('Loaded textual inversion captions!')
        #         self.textual_inversion_captions = json.load(f)
        #         # get max length
        #         self.textual_inversion_max_length = max([len(caption) for caption in self.textual_inversion_captions])
        #     for param in sd_model.cond_stage_model.parameters():
        #         param.requires_grad = False
        #     keep_cond = True
        if self.cfg['cross_blip_caption_path'] is not None:
            with open(self.cfg['cross_blip_caption_path'], 'r') as f:
                self.cross_blip_captions = json.load(f)
                # get max length
                self.cross_blip_max_length = max([len(caption) for caption in self.cross_blip_captions])
        else:
            self.cross_blip_captions = None

        if 'class_names' in self.cfg['text_conditioning']:
            self.class_names = self.cfg['class_names']
            with torch.no_grad():
                sd_model.cond_stage_model.to('cuda')
                class_emb_stack = []
                for class_name in self.class_names:
                    emb = sd_model.get_learned_conditioning(class_name)[[0], 1]
                    class_emb_stack.append(emb)
                self.class_names_embs = torch.stack(class_emb_stack, dim=1)

        # if not keep_cond:
        #     del sd_model.cond_stage_model

        self.use_decoder = use_decoder
        self.sd_model = sd_model

        # class embeddings & text adapter
        # TODO: implement me

        # check if class_embedding_path exists
        if not os.path.exists(class_embedding_path):
            print('No class embeddings provided!, please run create_class_embeddings.py --dataset pascal')

        class_embeddings = torch.load(class_embedding_path)
        self.register_buffer('class_embeddings', class_embeddings)
        text_dim = class_embeddings.size(-1)
        self.gamma = nn.Parameter(torch.ones(text_dim) * gamma_init_value)
        self.text_adapter = TextAdapter(text_dim=text_dim)

        # check if using the correct class embeddings
        # assert class_embeddings.size(0) == self.n_classes

        self.with_neck = True

        # num_classes = self.n_classes

        self.decode_head = nn.Module()
        enc_mid_channels, enc_end_channels = self.compute_decoder_head_shapes()
        backbone = torchvision.models.resnet50(pretrained=True)
        backbone.out_channels = 256

        # anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        anchor_sizes = ((32,), (64,), (128,), (256,),)
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                        output_size=7,
                                                        sampling_ratio=2)

        rnn = CustomFasterRCNN(backbone, num_classes=21,
                               rpn_anchor_generator=anchor_generator,
                               box_roi_pool=roi_pooler,
                               min_size=512)
        enc_mid_channels, enc_end_channels = self.compute_decoder_head_shapes()
        if hacky:
            enc_mid_channels += 19
            enc_end_channels += 19
        self.fpn = FeaturePyramidNetwork(in_channels_list=[320, enc_mid_channels, enc_end_channels, 1280],
                                         out_channels=256,
                                         extra_blocks=None,  # TODO:
                                         norm_layer=None)
        self.decode_head = rnn

        self.decode_head.num_classes = self.n_classes

        # self._init_decode_head(decode_head)
        # self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # assert self.with_decode_head

        ## test for now fix the unet
        # for param in self.unet.parameters():
        #     param.requires_grad = False

        if self.cfg["freeze_text_adapter"]:
            for param in self.text_adapter.parameters():
                param.requires_grad = False

        textual_inversion_token_path = self.cfg['textual_inversion_token_path']
        if textual_inversion_token_path is not None:
            self.text_encoder = sd_model.cond_stage_model.transformer
            self.tokenizer = sd_model.cond_stage_model.tokenizer
            self._load_textual_inversion_token(textual_inversion_token_path)

    def init_evaluator(self):
        self.evaluator = PascalVOCDetectionEvaluator(dataset_name='voc_2007_test')
        if self.dataset_name == "watercolor":
            self.evaluator._anno_file_template = './data/cross-domain-detection/datasets/watercolor/Annotations/{}.xml'
            self.evaluator._image_set_path = './data/cross-domain-detection/datasets/watercolor/ImageSets/Main/test.txt'
            self.evaluator._class_names = REDUCED_CLASS_NAMES[1:]
            self.evaluator._is_2007 = True
        elif self.dataset_name == "comic":
            self.evaluator._anno_file_template = './data/cross-domain-detection/datasets/comic/Annotations/{}.xml'
            self.evaluator._image_set_path = './data/cross-domain-detection/datasets/comic/ImageSets/Main/test.txt'
            self.evaluator._class_names = REDUCED_CLASS_NAMES[1:]
            self.evaluator._is_2007 = True
            print('d')
        self.evaluator.reset()

    def _load_textual_inversion_token(self, token):
        token_ids_and_embeddings = []

        # 1. Load textual inversion file
        model_file = _get_model_file(
            token,
            weights_name="learned_embeds.bin",
            cache_dir=DIFFUSERS_CACHE,
            force_download=False,
            resume_download=False,
            proxies=None,
            local_files_only=False,
            use_auth_token=None,
            revision=None,
            subfolder=None,
            user_agent={"file_type": "text_inversion", "framework": "pytorch"},
        )
        state_dict = torch.load(model_file, map_location="cpu")

        # Save text_encoder and tokenizer

        # 2. Load token and embedding correctly from file
        loaded_token, embedding = next(iter(state_dict.items()))
        token = loaded_token
        embedding = embedding.to(dtype=self.text_encoder.dtype, device=self.text_encoder.device)

        # 3. Make sure we don't mess up the tokenizer or text encoder
        vocab = self.tokenizer.get_vocab()
        if token in vocab:
            raise ValueError(
                f"Token {token} already in tokenizer vocabulary. Please choose a different token name or remove {token} and embedding from the tokenizer and text encoder."
            )

        tokens = [token]
        embeddings = [embedding[0]] if len(embedding.shape) > 1 else [embedding]

        # add tokens and get ids
        self.tokenizer.add_tokens(tokens)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids_and_embeddings += zip(token_ids, embeddings)

        print(f"Loaded textual inversion embedding for {token}.")

        # resize token embeddings and set all new embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        for token_id, embedding in token_ids_and_embeddings:
            self.text_encoder.get_input_embeddings().weight.data[token_id] = embedding

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

        if ('blip' in text_cond) or ('textual_inversion' in text_cond):
            enc_mid_channels += 77
            enc_end_channels += 77

        if 'class_names' in text_cond:
            enc_mid_channels += len(self.class_names)
            enc_end_channels += len(self.class_names)

        if 'class_emb' in text_cond:
            enc_mid_channels += self.n_classes
            enc_end_channels += self.n_classes

        return enc_mid_channels, enc_end_channels

    def create_text_embeddings(self, latents, img_metas=None, captions=None):
        text_cond = self.cfg['text_conditioning']
        conds = []
        if 'blip' in text_cond:
            texts = []
            _cs = []
            for img_id in img_metas['img_id']:
                if self.training or self.cfg['val_dataset_name'] == 'pascal':
                    text = captions[img_id]['captions']
                else:
                    text = self.cross_blip_captions[img_id]['captions']
                if len(text) > 1:  # TODO: hacky, rm me later?
                    text = text[0]
                if ('train' in text_cond and self.training) or not self.training:
                    if 'watercolor' in text_cond:
                        text = "a watercolor painting of a " + text[0]
                        text = [text]
                    if 'dashcam' in text_cond:
                        text = "a dashcam photo of a " + text[0]
                        text = [text]
                    if 'constructive' in text_cond:
                        text = "a constructivism painting of a " + text[0]
                        text = [text]
                    if 'watercolorTI' in text_cond:
                        text = "a <watercolor> style painting of a " + text[0]
                        text = [text]
                    if 'watercolorDB' in text_cond:
                        text = "a sks style painting of a " + text[0]
                        text = [text]
                    if 'comic' in text_cond:
                        text = "a comic style painting of a " + text[0]
                        text = [text]
                    if 'comicTI' in text_cond:
                        text = "a <comic> style painting of a " + text[0]
                        text = [text]
                    if 'comicDB' in text_cond:
                        text = "a sks style painting of a " + text[0]
                        text = [text]
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
            c_crossattn = self.text_adapter(latents, self.class_embeddings,
                                            self.gamma)  # NOTE: here the c_crossattn should be expand_dim as latents
            conds.append(c_crossattn)

        c_crossattn = torch.cat(conds, dim=1)
        return c_crossattn

    def extract_feat(self, img, img_metas=None):
        if img_metas['img_id'][0] == '2011_002362':
            print(';')
        """Extract features from images."""
        if self.cfg.get('use_scaled_encode', False):
            with torch.no_grad():
                latents = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(img))
        else:
            with torch.no_grad():
                latents = self.encoder_vq.encode(img)
            latents = latents.mode().detach()

        # TODO: be careful, hacky zone!!!!!
        viz = False

        blip_captions = None
        if 'blip' in self.cfg['text_conditioning']:
            blip_captions = self.blip_captions

        c_crossattn = self.create_text_embeddings(latents, img_metas=img_metas, captions=blip_captions)

        t = torch.from_numpy(np.array([1])).to(img.device)

        outs = self.model(latents, t, c_crossattn=[c_crossattn])

        if hacky:
            for i in range(self.class_embeddings.size(0) - 2):
                c_crossattn = self.text_adapter(latents,
                                                self.class_embeddings[i:i + 2],
                                                # self.class_embeddings,
                                                self.gamma)  # NOTE: here the c_crossattn should be expand_dim as latents

                # t = torch.ones((img.shape[0],), device=img.device).long()
                # more timesteps
                t = torch.from_numpy(np.array([1])).to(img.device)

                per_layer_outs = self.model(latents, t, c_crossattn=[c_crossattn])
                for layerOuts_idx, layerOuts in enumerate(per_layer_outs):
                    if layerOuts_idx == 0 or layerOuts_idx == 3:
                        continue
                    outs[layerOuts_idx] = torch.cat((outs[layerOuts_idx], layerOuts[:, -2:-1, :, :]), dim=1)

            print('h')

        if viz:
            # vis
            plt.imshow(outs[1][0][1].detach().cpu().numpy())
            plt.show()
            #
            plt.imshow(outs[1][0][-2].detach().cpu().numpy())
            plt.show()

        return outs

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

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

    def forward(self, x, cross_target=None, img_metas=None, targets=None):
        x = [torchvision.transforms.ToTensor()(x) for x in x]
        y = targets
        # just imagenet normalization
        orig_images = [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])(x) for x in x]
        orig_images = [datapoints.Image(x) for x in orig_images]
        # resize to 800x800
        bboxes = [datapoints.BoundingBox(y['boxes'], format="XYXY", spatial_size=orig_images[i].shape[1:]) for i, y
                  in
                  enumerate(y)]
        from torchvision.transforms import v2 as T
        from torchvision.transforms.v2 import functional as F
        _size = 512
        trans = T.Compose(
            [
                T.Resize((_size, _size)),
            ]
        )
        trans([orig_images[0]], [bboxes[0]])
        a = [trans(orig_images[i], bboxes[i]) for i in range(len(orig_images))]
        orig_images = [a[0] for a in a]
        bboxes = [a[1] for a in a]
        for i, _ in enumerate(y):
            y[i]['boxes'] = bboxes[i].clone().detach()

        orig_images = torch.stack(orig_images)
        orig_images = torchvision.models.detection.image_list.ImageList(orig_images,
                                                                        image_sizes=[(_size, _size)] * len(orig_images))

        orig_images_tensors = orig_images.tensors.to(self.device)
        features = self.extract_feat(orig_images_tensors, img_metas=img_metas)

        feat_names = ['0', '1', '2', '3']
        features = dict(zip(feat_names, features))

        if self.with_neck:
            features = self.fpn(features)
            x = self.decode_head(images=orig_images, images_tensor=orig_images_tensors, features=features,
                                 targets=targets)

        return x, y, orig_images.tensors

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img, img_metas)
        if self.with_neck:
            x = list(self.neck(x))
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        if torch.isnan(seg_logit).any():
            print('########### find NAN #############')

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3,))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2,))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred

    def initialize_loss(self):
        loss = smp.losses.FocalLoss(mode="multiclass", ignore_index=self.ignore_index)
        return loss

    def configure_optimizers(self):
        # TODO: double check here
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        # have differernt learning rate for different layers
        # parameters to optimize
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
            else:
                decay.append(m)

        if self.freeze_backbone:
            params_to_optimize = [
                {'params': lesslr_no_decay, 'weight_decay': 0.0, 'lr_scale': 0.0},
                {'params': lesslr_decay, 'lr_scale': 0.0},
                {'params': no_lr, 'lr_scale': 0.0},
                {'params': no_decay, 'weight_decay': 0.0},
            ]
        else:
            params_to_optimize = [
                {'params': lesslr_no_decay, 'weight_decay': 0.0, 'lr_scale': 0.01},
                {'params': lesslr_decay, 'lr_scale': 0.01},
                {'params': no_lr, 'lr_scale': 0.0},
                {'params': no_decay, 'weight_decay': 0.0},
                {'params': decay}
            ]
        optimizer = torch.optim.AdamW(params_to_optimize,
                                      lr=0.00001,
                                      # lr=0.000005,
                                      weight_decay=1e-2,
                                      amsgrad=False
                                      )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: (1 - x / (
                self.cfg["dataset_len"] * self.cfg["max_epochs"])) ** 0.9)

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch", "frequency": 1}]

    def training_step(self, batch, batch_idx, do_augmentation=False):
        images, annotation = zip(*batch)
        img_ids = [
            annot['annotation']['filename'].split('.jpg')[0]
            for
            annot in annotation]
        img_metas = {}
        img_metas['img_id'] = img_ids

        gt_boxes, gt_labels, gt_scores = annotations_to_boxes(annotation)

        if do_augmentation:

            from torchvision.transforms import v2 as v2_transforms

            transform = v2_transforms.Compose([
                v2_transforms.RandomHorizontalFlip(p=0.5),
                v2_transforms.RandomCrop(size=[800, 800], pad_if_needed=True),
            ])

            new_images = []
            new_labels = []
            for img, label in zip(images, gt_boxes):
                img_datapoint = datapoints.Image(img)
                label_datapoint = datapoints.BoundingBox(label, format="XYXY", spatial_size=img_datapoint.shape[1:])
                img_datapoint, label_datapoint = transform(img_datapoint, label_datapoint)

                new_images.append(img_datapoint)
                new_labels.append(label_datapoint)

            images = [v2_transforms.ToPILImage()(img) for img in new_images]
            gt_boxes = new_labels

            # check for any boxes that are completely outside the image and remove them (or at the edge) in xyxy format
            gt_boxes = [box for box in gt_boxes if box[0] > 0 and box[1] > 0 and box[2] > 0 and box[3] > 0]

        targets = []
        for i in range(len(gt_boxes)):
            target = {
                "image_id": i,
                "boxes": gt_boxes[i].to(self.device),  # (n_objects, 4)
                "labels": gt_labels[i].to(self.device),  # (n_objects)
            }
            targets.append(target)
        y = targets
        losses, _, _ = self(images, targets=y, img_metas=img_metas)
        # TODO: check in detectron engine how the losses are used
        total_loss = sum(losses.values())
        self.log("train_loss", total_loss)

        del images
        del annotation
        del img_ids
        del img_metas
        del gt_boxes
        del gt_labels
        del gt_scores
        del targets
        del y
        del losses

        # torch.cuda.empty_cache()

        return total_loss

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path, strict=True):
        # TODO diff be
        self.load_state_dict(torch.load(path)['state_dict'], strict=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0, plotting=False):
        try:
            images, gt_boxes, gt_labels, annotation = zip(*batch)
            img_ids = [
                str(annot['image_id'])
                for
                annot in annotation]
            img_metas = {}
            img_metas['img_id'] = img_ids
            targets = []
            for i in range(len(gt_boxes)):
                target = {
                    "image_id": img_metas['img_id'][i],
                    "boxes": gt_boxes[i].to(self.device),  # (n_objects, 4)
                    "labels": gt_labels[i].to(self.device),  # (n_objects)
                }
                targets.append(target)
            y = targets
            images = [torchvision.transforms.ToPILImage()(x) for x in images]
        except:
            images, annotation = zip(*batch)

            img_ids = [
                annot['annotation']['filename'].split('.jpg')[0]
                for
                annot in annotation]
            img_metas = {}
            img_metas['img_id'] = img_ids

            gt_boxes, gt_labels, gt_scores = annotations_to_boxes(annotation)

            targets = []
            for i in range(len(gt_boxes)):
                target = {
                    "image_id": img_metas['img_id'][i],
                    "boxes": gt_boxes[i].to(self.device),  # (n_objects, 4)
                    "labels": gt_labels[i].to(self.device),  # (n_objects)
                }
                targets.append(target)
            y = targets

        results, resized_targets, resized_imgs = self(images, targets=y, img_metas=img_metas)
        # TODO: check in detectron engine how the losses are used
        pred_boxes = [x["boxes"] for x in results]
        pred_labels = [x["labels"] for x in results]
        pred_scores = [x["scores"] for x in results]

        use_wandb = True

        original_shapes = [x.size for x in images]
        resized_shapes = [x.shape[1:] for x in resized_imgs]
        resizing_ratios = [(x[0] / y[0], x[1] / y[1]) for x, y in zip(original_shapes, resized_shapes)]

        # resize boxes to original size based on resizing ratios
        for i in range(len(pred_boxes)):
            pred_boxes[i][:, 0] *= resizing_ratios[i][0]
            pred_boxes[i][:, 1] *= resizing_ratios[i][1]
            pred_boxes[i][:, 2] *= resizing_ratios[i][0]
            pred_boxes[i][:, 3] *= resizing_ratios[i][1]

        preds = []
        for i in range(len(pred_boxes)):
            boxes = pred_boxes[i]
            labels = pred_labels[i]
            scores = pred_scores[i]
            orig_shapes = original_shapes[i]

            if self.dataset_name == "watercolor" or self.dataset_name == "comic":
                labels_as_words = [DETECTRON_VOC_CLASS_NAMES[l] for l in labels]

                #
                # remap labels from DETECTRON_VOC_CLASS_NAMES to REDUCED_CLASS_NAMES
                new_labels = []
                skipped_label_idx = []
                for idx, label in enumerate(labels_as_words):
                    try:
                        new_label = REDUCED_CLASS_NAMES.index(label)
                        new_labels.append(new_label)
                    except:
                        skipped_label_idx.append(idx)
                labels = torch.tensor(new_labels)
                # remove skipped boxes
                boxes = [boxes[i] for i in range(len(boxes)) if i not in skipped_label_idx]
                scores = [scores[i] for i in range(len(scores)) if i not in skipped_label_idx]
                if len(boxes) > 0:
                    scores = torch.tensor(scores)
                    boxes = torch.stack(boxes)
                else:
                    scores = torch.tensor([])
                    boxes = torch.tensor([])

            labels = labels.tolist()

            # pred = {"boxes": boxes, "labels": labels, "scores": scores, "image_id": torch.tensor([i])}
            pred = {"instances": Instances(image_size=orig_shapes, pred_boxes=boxes, pred_classes=labels,
                                           scores=scores)}
            preds.append(pred)
        gt = []
        for i in range(len(gt_boxes)):
            # gt.append({"boxes": gt_boxes[i], "labels": gt_labels[i]})
            gt_labels_as_words = resized_targets[i]["labels"].tolist()
            gt.append({
                # "boxes": resized_targets[i]["boxes"],
                "boxes": gt_boxes[i],
                "labels": gt_labels_as_words,
                "image_id": resized_targets[i]["image_id"]
            })

        # use detectron2  evaluator
        self.evaluator.process(gt, preds)

        # prep data for torchmetrics COCO API
        # remove 'image_id' from gt
        for i in range(len(gt)):
            del gt[i]['image_id']

        preds = []
        for i in range(len(pred_boxes)):
            boxes = pred_boxes[i]
            labels = pred_labels[i]
            scores = pred_scores[i]

            pred = {"boxes": boxes, "labels": labels, "scores": scores, "image_id": torch.tensor([i])}
            preds.append(pred)

        gt = []
        for i in range(len(gt_boxes)):
            gt.append({"boxes": gt_boxes[i], "labels": gt_labels[i]})
            # gt.append({"boxes": resized_targets[i]["boxes"], "labels": resized_targets[i]["labels"], })

        gt = [{k: v.to(self.device) for k, v in t.items()} for t in gt]
        preds = [{k: v.to(self.device) for k, v in t.items()} for t in preds]

        res = self.metric(preds, gt)
        self.log("val_mAP_{}".format(self.dataset_name), res['map'], on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=len(batch))
        self.log("val_mAP_50_{}".format(self.dataset_name), res['map_50'], on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=len(batch))

        self.all_predictions.append(preds)
        self.all_targets.append(gt)

        # for the first 5 images, save the images with bounding boxes using the torchvision function utils
        # in folder ./object_detection_results/
        res_path = "./object_detection_results/"
        if not os.path.exists(res_path):
            os.makedirs(res_path)

        if batch_idx < 10 and plotting:
            for i, (img, result, gt) in enumerate(zip(images, results, gt)):
                try:
                    img = img.permute(1, 2, 0).cpu().numpy()
                except:
                    img = torchvision.transforms.ToTensor()(img).permute(1, 2, 0).cpu().numpy()
                # denormalize from imagenet mean and std
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3).cpu().numpy()
                img = img + torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3).cpu().numpy()
                img = img * 255
                img = img.astype(np.uint8)
                img_pil = Image.fromarray(img)

                # boxes = result["boxes"]
                # labels = result["labels"]
                # scores = result["scores"]
                boxes = pred_boxes[i]
                labels = pred_labels[i]
                scores = pred_scores[i]
                filtered_boxes = boxes[scores > 0.5]
                filtered_labels = labels[scores > 0.5]

                img_tensor = torchvision.transforms.functional.to_tensor(img_pil)

                # Plot predicted bounding boxes
                fig, ax = plt.subplots()
                ax.imshow(torchvision.transforms.functional.to_pil_image(img_tensor))

                if not filtered_boxes.shape[0] == 0:

                    for box, label in zip(filtered_boxes, filtered_labels):
                        xmin, ymin, xmax, ymax = box.tolist()
                        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='red',
                                             linewidth=2)
                        ax.add_patch(rect)
                        ax.text(xmin, ymin, label.item(), bbox=dict(facecolor='red', alpha=0.5))

                plt.axis('off')

                # Save or log predicted image
                if use_wandb:
                    self.logger.experiment.log({"predicted_image": wandb.Image(fig)})
                else:
                    plt.savefig(f"./object_detection_results/epoch_{self.current_epoch}_{batch_idx}_{i}_pred.jpg",
                                bbox_inches='tight', pad_inches=0)

                boxes = gt["boxes"].to(torch.int64)
                labels = gt["labels"].to(torch.int64)

                # Plot ground truth bounding boxes
                fig, ax = plt.subplots()
                ax.imshow(torchvision.transforms.functional.to_pil_image(img_tensor))

                for box, label in zip(boxes, labels):
                    xmin, ymin, xmax, ymax = box.tolist()
                    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='green',
                                         linewidth=2)
                    ax.add_patch(rect)
                    ax.text(xmin, ymin, label.item(), bbox=dict(facecolor='green', alpha=0.5))

                plt.axis('off')

                # Save or log ground truth image
                if use_wandb:
                    self.logger.experiment.log({"ground_truth_image": wandb.Image(fig)})
                else:
                    plt.savefig(f"./object_detection_results/epoch_{self.current_epoch}_{batch_idx}_{i}_gt.jpg",
                                bbox_inches='tight', pad_inches=0)

        # delete stuff
        del gt
        del preds
        del results
        del y
        del gt_boxes
        del gt_labels
        try:
            del gt_scores
        except:
            pass
        del pred_boxes
        del pred_labels
        del pred_scores
        del img_ids
        del img_metas
        del images
        del annotation
        del resized_targets
        del resized_imgs

    def on_validation_epoch_end(self):
        # do full validation
        # self.validation_step(self.val_dataloader())
        print('validation epoch end')
        self.evaluator._is_2007 = True
        res = self.evaluator.evaluate()
        bbox_aps = res['bbox']
        self.log("pascal_eval_2007_val_mAP_{}".format(self.dataset_name), bbox_aps['AP'], on_step=False, on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log("pascal_eval_2007_val_mAP_50_{}".format(self.dataset_name), bbox_aps['AP50'], on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log("pascal_eval_2007_val_mAP_75_{}".format(self.dataset_name), bbox_aps['AP75'], on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        self.evaluator._is_2007 = False
        res = self.evaluator.evaluate()
        bbox_aps = res['bbox']
        self.log("pascal_eval_2012_val_mAP_{}".format(self.dataset_name), bbox_aps['AP'], on_step=False, on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log("pascal_eval_2012_val_mAP_50_{}".format(self.dataset_name), bbox_aps['AP50'], on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log("pascal_eval_2012_val_mAP_75_{}".format(self.dataset_name), bbox_aps['AP75'], on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        self.evaluator.reset()

        use_chainercv = False
        if use_chainercv:
            # also eval using chainercv
            # prepare nd arrays for eval with chainercv
            all_pred_boxes = []
            all_pred_labels = []
            all_pred_scores = []
            all_gt_boxes = []
            all_gt_labels = []
            for i in range(len(self.all_predictions)):
                pred_boxes = self.all_predictions[i][0]['boxes'].cpu().numpy()
                pred_labels = self.all_predictions[i][0]["labels"].cpu().numpy()
                pred_scores = self.all_predictions[i][0]["scores"].cpu().numpy()

                all_pred_boxes.append(pred_boxes)
                all_pred_labels.append(pred_labels)
                all_pred_scores.append(pred_scores)

                gt_boxes = self.all_targets[i][0]["boxes"].cpu().numpy()
                gt_labels = self.all_targets[i][0]["labels"].cpu().numpy()

                all_gt_boxes.append(gt_boxes)
                all_gt_labels.append(gt_labels)

            # save all predictions and targets to disk as pickle
            with open("all_pred_boxes.pkl", "wb") as f:
                pickle.dump(all_pred_boxes, f)
            with open("all_pred_labels.pkl", "wb") as f:
                pickle.dump(all_pred_labels, f)
            with open("all_pred_scores.pkl", "wb") as f:
                pickle.dump(all_pred_scores, f)
            with open("all_gt_boxes.pkl", "wb") as f:
                pickle.dump(all_gt_boxes, f)
            with open("all_gt_labels.pkl", "wb") as f:
                pickle.dump(all_gt_labels, f)

            res = eval_detection_voc(all_pred_boxes, all_pred_labels, all_pred_scores, all_gt_boxes, all_gt_labels,
                                     use_07_metric=True)

            self.log("chainercv_val_mAP_{}".format(self.dataset_name), res['map'], on_step=False, on_epoch=True,
                     prog_bar=True,
                     logger=True)
            self.log("chainercv_val_AP_{}".format(self.dataset_name), res['ap'], on_step=False, on_epoch=True,
                     prog_bar=True,
                     logger=True)

            res = eval_detection_voc(self.all_predictions, self.all_targets, use_07_metric=True)
            self.log("chainercv_val_mAP_2007_{}".format(self.dataset_name), res['map'], on_step=False, on_epoch=True,
                     prog_bar=True,
                     logger=True)
            self.log("chainercv_val_AP_2007_{}".format(self.dataset_name), res['ap'], on_step=False, on_epoch=True,
                     prog_bar=True,
                     logger=True)

        self.all_predictions = []
        self.all_targets = []

        pass

    def validation_step_object_detection(self, batch, batch_idx):
        x = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]
        metas = [x[3] for x in batch]
        new_metas = {
            'img_id': [x['image_id'] for x in metas],
            'img_path': [x['image_path'] for x in metas],
        }
        metas = new_metas
        targets = []
        for i in range(len(boxes)):
            target = {
                "image_id": torch.tensor([i]),
                "boxes": boxes[i],
                "labels": labels[i],
            }
            targets.append(target)
        y = targets
        x = torch.stack(x)
        results = self(x, cross_target='watercolor', img_metas=metas)
        # calulate metrics for validation
        # argmax for masks
        results = torch.argmax(results, dim=1)
        a = create_bounding_boxes_from_masks(results)
        gt_boxes = [x["boxes"] for x in y]
        gt_labels = [x["labels"] for x in y]
        # pred_boxes = [x["boxes"] for x in results]
        # pred_labels = [x["labels"] for x in results]
        # pred_scores = [x["scores"] for x in results]
        # empty tensor
        pred_boxes = []
        pred_labels = []
        for i in range(len(results)):
            pred_boxes_for_image = []
            pred_labels_for_image = []
            for label, box in a[i]:
                pred_boxes_for_image.append(box.to(self.device))
                pred_labels_for_image.append(label.to(self.device))

            pred_boxes.append(pred_boxes_for_image)
            pred_labels.append(pred_labels_for_image)

        use_wandb = True

        # calculate mAP

        # self.metric.update(pred_boxes, pred_labels, gt_boxes, gt_labels)
        # self.log("val_mAP", self.metric.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #
        # mAP(pred_boxes, pred_labels, gt_boxes, gt_labels)
        # self.log("val_mAP", mAP, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # return mAP

        ###  --- version for pytorchmodels
        # preds = []
        # for i in range(len(pred_boxes)):
        #     boxes = pred_boxes[i]
        #     labels = pred_labels[i]
        #     # scores = pred_scores[i]
        #
        #     # pred = {"boxes": boxes, "labels": labels, "scores": scores}
        #     pred = {"boxes": boxes, "labels": labels}
        #     preds.append(pred)
        # gt = []
        # for i in range(len(gt_boxes)):
        #     gt.append({"boxes": gt_boxes[i], "labels": gt_labels[i]})

        # ### --- version for torchmetrics
        # preds = []
        # for i in range(len(pred_boxes)):
        #     boxes = pred_boxes[i]
        #     labels = pred_labels[i]
        #     # scores = pred_scores[i]
        #
        #     # pred = {"boxes": boxes, "labels": labels, "scores": scores}
        #     try:
        #         pred = {"boxes": torch.stack(boxes), "labels": torch.stack(labels)}
        #     except:
        #         print("no boxes")
        #         pred = {"boxes": torch.ones([0, 4]), "labels": torch.ones([0])}
        #     preds.append(pred)
        # gt = []
        # for i in range(len(gt_boxes)):
        #     gt.append({"boxes": gt_boxes[i], "labels": gt_labels[i]})

        metric = MeanAveragePrecision()
        for pred_box, pred_label, gt_box, gt_label in zip(pred_boxes, pred_labels, gt_boxes, gt_labels):
            try:
                _preds = [
                    dict(
                        boxes=torch.stack(pred_box).to(self.device),
                        labels=torch.stack(pred_label).to(self.device),
                        scores=torch.ones(len(pred_box)).to(self.device),
                    )
                ]
                _gt = [
                    dict(
                        boxes=gt_box.to(self.device),
                        labels=gt_label.to(self.device),
                    )
                ]
                metric.update(_preds, _gt)
            except:
                print("no boxes")
                _preds = [
                    dict(
                        boxes=torch.ones([0, 4]).to(self.device),
                        labels=torch.ones([0]).to(self.device),
                        scores=torch.ones([0]).to(self.device),
                    )
                ]
                _gt = [
                    dict(
                        boxes=gt_box.to(self.device),
                        labels=gt_label.to(self.device),
                    )
                ]
                metric.update(_preds, _gt)

        res = metric.compute()

        # self.log("val_mAP_{}".format(self.dataset_name), res['map'], on_step=False, on_epoch=True, prog_bar=True,
        #          logger=True)
        # self.log("val_mAP_50_{}".format(self.dataset_name), res['map_50'], on_step=False, on_epoch=True, prog_bar=True,
        #          logger=True)
        self.log("val_mAP_watercolor", res['map'], on_step=False, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log("val_mAP_50_watercolor", res['map_50'], on_step=False, on_epoch=True, prog_bar=True,
                 logger=True)

        # for the first 5 images, save the images with bounding boxes using the torchvision function utils
        # in folder ./object_detection_results/
        res_path = "./object_detection_results/"
        if not os.path.exists(res_path):
            os.makedirs(res_path)

        if batch_idx < 10:
            for i, (img, result, gt) in enumerate(zip(x, results, y)):
                img = img.permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                img_pil = Image.fromarray(img)

                boxes = pred_boxes[i]
                labels = pred_labels[i]
                # scores = torch.ones(len(boxes))
                # filtered_boxes = boxes[scores > 0.5]
                # filtered_labels = labels[scores > 0.5]
                filtered_boxes = boxes
                filtered_labels = labels

                img_tensor = torchvision.transforms.functional.to_tensor(img_pil)

                # Plot predicted bounding boxes
                fig, ax = plt.subplots()
                ax.imshow(torchvision.transforms.functional.to_pil_image(img_tensor))

                for box, label in zip(filtered_boxes, filtered_labels):
                    xmin, ymin, xmax, ymax = box.tolist()
                    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='red',
                                         linewidth=2)
                    ax.add_patch(rect)
                    ax.text(xmin, ymin, label.item(), bbox=dict(facecolor='red', alpha=0.5))

                plt.axis('off')

                # Save or log predicted image
                if use_wandb:
                    self.logger.experiment.log({"predicted_image": wandb.Image(fig)})
                else:
                    plt.savefig(f"./object_detection_results/epoch_{self.current_epoch}_{batch_idx}_{i}_pred.jpg",
                                bbox_inches='tight', pad_inches=0)

                boxes = gt["boxes"].to(torch.int64)
                labels = gt["labels"].to(torch.int64)

                # Plot ground truth bounding boxes
                fig, ax = plt.subplots()
                ax.imshow(torchvision.transforms.functional.to_pil_image(img_tensor))

                for box, label in zip(boxes, labels):
                    xmin, ymin, xmax, ymax = box.tolist()
                    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='green',
                                         linewidth=2)
                    ax.add_patch(rect)
                    ax.text(xmin, ymin, label.item(), bbox=dict(facecolor='green', alpha=0.5))

                plt.axis('off')

                # Save or log ground truth image
                if use_wandb:
                    self.logger.experiment.log({"ground_truth_image": wandb.Image(fig)})
                else:
                    plt.savefig(f"./object_detection_results/epoch_{self.current_epoch}_{batch_idx}_{i}_gt.jpg",
                                bbox_inches='tight', pad_inches=0)

        return res
