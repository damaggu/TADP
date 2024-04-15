import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor

import sys
from omegaconf import OmegaConf
from einops import rearrange, repeat

from ldm.util import instantiate_from_config
from TADP.vpd import UNetWrapper, TextAdapter
import json


@SEGMENTORS.register_module()
class TADPSeg(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 decode_head,
                 sd_path='checkpoints/v1-5-pruned-emaonly.ckpt',
                 unet_config=dict(),
                 class_embedding_path='TADP/vpd/ade_class_embeddings.pth',
                 gamma_init_value=1e-4,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 ## added by us
                 use_decoder=False,
                 opt_dict=None,
                 ckpt_path=None,

                 **args):
        super().__init__(init_cfg)
        self.opt_dict = opt_dict
        config = OmegaConf.load('./stable_diffusion/configs/stable-diffusion/v1-inference.yaml')

        # Hard coding ckpt path for local testing, so don't have to save multiple copies of model
        if ckpt_path is not None:
            config.model.params.ckpt_path = ckpt_path
        else:
            config.model.params.ckpt_path = f'./{sd_path}'

        if (('blip' not in self.opt_dict["text_conditioning"]) and (
                'textual_inversion' not in self.opt_dict["text_conditioning"]) and
                (self.opt_dict["text_conditioning"] != 'prompt_input')):
            config.model.params.cond_stage_config.target = 'ldm.modules.encoders.modules.AbstractEncoder'

        #####

        # prepare the unet
        sd_model = instantiate_from_config(config.model)
        if self.opt_dict['dreambooth_checkpoint'] is not None:
            sd_model.load_state_dict(torch.load(self.opt_dict['dreambooth_checkpoint'])['state_dict'], strict=False)
            print('Loaded dreambooth checkpoint!')

        # handle logic for using scaled encoder
        if not self.opt_dict.get('use_scaled_encode', False):
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

        self.unet = UNetWrapper(sd_model.model, **unet_config)

        sd_model.model = None
        text_dim = None
        _tc = self.opt_dict["text_conditioning"]
        if 'blip' in _tc or _tc == 'prompt_input':
            if 'blip' in _tc:
                with open(self.opt_dict['blip_caption_path'], 'r') as f:
                    print('Loaded blip captions!')
                    self.blip_captions = json.load(f)
                    # get max length
                    self.blip_max_length = max([len(caption) for caption in self.blip_captions])
            for param in sd_model.cond_stage_model.parameters():
                param.requires_grad = False

            if self.opt_dict['cross_blip_caption_path'] is not None:
                self.cross_blip_captions = {}
                for path in self.opt_dict['cross_blip_caption_path'].split(','):
                    with open(path, 'r') as f:
                        print('Loaded cross blip captions!')
                        cross_blip_captions = json.load(f)
                        # get max length
                        self.cross_blip_max_length = max([len(caption) for caption in cross_blip_captions])
                        self.cross_blip_captions.update(cross_blip_captions)
            else:
                self.cross_blip_captions = None
            text_dim = 768
        # if 'textual_inversion' in self.opt_dict["text_conditioning"]:
        # with open(self.opt_dict['textual_inversion_caption_path'], 'r') as f:
        #     print('Loaded textual inversion captions!')
        #     self.textual_inversion_captions = json.load(f)
        #     get max length
        # self.textual_inversion_max_length = max([len(caption) for caption in self.textual_inversion_captions])
        # for param in sd_model.cond_stage_model.parameters():
        #     param.requires_grad = False
        # text_dim = 768
        _tc = self.opt_dict["text_conditioning"]
        if not 'blip' in _tc and not 'textual_inversion' in _tc and not 'prompt_input' in _tc:
            del sd_model.cond_stage_model

        self.use_decoder = use_decoder
        self.sd_model = sd_model

        ####
        # class embeddings & text adapter
        if 'class_emb' in self.opt_dict["text_conditioning"]:
            class_embeddings = torch.load(class_embedding_path)
            self.register_buffer('class_embeddings', class_embeddings)
            text_dim = class_embeddings.size(-1)

        if self.opt_dict['use_text_adapter']:
            assert text_dim is not None
            self.text_adapter = TextAdapter(text_dim=text_dim)
            self.gamma = nn.Parameter(torch.ones(text_dim) * gamma_init_value)

        self.num_classes = decode_head['num_classes']
        enc16_size, enc32_size = self.compute_decoder_head_shapes()
        neck['in_channels'][1] = enc16_size
        neck['in_channels'][2] = enc32_size

        if neck is not None:
            self.neck = builder.build_neck(neck)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert self.with_decode_head

        textual_inversion_token_path = self.opt_dict['textual_inversion_token_path']
        if textual_inversion_token_path is not None:
            # self.text_encoder = sd_model.cond_stage_model.transformer
            # self.tokenizer = sd_model.cond_stage_model.tokenizer
            self._load_textual_inversion_token(textual_inversion_token_path)

    @property
    def device(self):
        return self.sd_model.cond_stage_model.transformer.device

    def _load_textual_inversion_token(self, token):
        token_ids_and_embeddings = []
        from diffusers.utils import _get_model_file, DIFFUSERS_CACHE

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
        embedding = embedding.to(dtype=self.sd_model.cond_stage_model.transformer.dtype, device=self.device)

        # 3. Make sure we don't mess up the tokenizer or text encoder
        vocab = self.sd_model.cond_stage_model.tokenizer.get_vocab()
        if token in vocab:
            raise ValueError(
                f"Token {token} already in tokenizer vocabulary. Please choose a different token name or remove {token} and embedding from the tokenizer and text encoder."
            )

        tokens = [token]
        embeddings = [embedding[0]] if len(embedding.shape) > 1 else [embedding]

        # add tokens and get ids
        self.sd_model.cond_stage_model.tokenizer.add_tokens(tokens)
        token_ids = self.sd_model.cond_stage_model.tokenizer.convert_tokens_to_ids(tokens)
        token_ids_and_embeddings += zip(token_ids, embeddings)

        print(f"Loaded textual inversion embedding for {token}.")

        # resize token embeddings and set all new embeddings
        self.sd_model.cond_stage_model.transformer.resize_token_embeddings(
            len(self.sd_model.cond_stage_model.tokenizer))
        for token_id, embedding in token_ids_and_embeddings:
            self.sd_model.cond_stage_model.transformer.get_input_embeddings().weight.data[token_id] = embedding

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

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
        text_cond = self.opt_dict['text_conditioning'].split('+')
        enc_16_channels = 640
        enc_32_channels = 1280
        if self.opt_dict['append_self_attention']:
            enc_16_channels += 1024
            enc_32_channels += 256

        if ('blip' in text_cond) or ('textual_inversion' in text_cond) or 'prompt_input' in text_cond:
            enc_16_channels += 77
            enc_32_channels += 77

        if 'class_emb' in text_cond:
            enc_16_channels += self.num_classes
            enc_32_channels += self.num_classes

        return enc_16_channels, enc_32_channels

    def create_text_embeddings(self, img_metas, latents):
        bsz = len(latents)
        text_cond = self.opt_dict['text_conditioning'].split('+')
        conds = []

        if 'prompt_input' in text_cond:
            custom_prompts = [meta['prompt'] for meta in img_metas]
            assert all([_p is not None for _p in custom_prompts]), "Prompt input requires custom prompts"
            if isinstance(custom_prompts, str):
                custom_prompts = [custom_prompts]
            conds = []
            texts = []
            _cs = []
            for text in custom_prompts:
                c = self.sd_model.get_learned_conditioning([text])
                texts.append(text)
                _cs.append(c)
            c = torch.cat(_cs, dim=0)
            conds.append(c)

        if 'blip' in text_cond:
            texts = []
            _cs = []
            for img_meta in img_metas:
                img_id = img_meta['ori_filename']
                ### use cross domain captions during testing
                if self.cross_blip_captions is not None:
                    if self.training:
                        text = self.blip_captions[img_id]['captions']
                    else:
                        text = self.cross_blip_captions[img_id]['captions']
                    if ('train' in text_cond and self.training) or not self.training:
                        if 'condition' in text_cond:
                            text = 'a ' + "dark night" + ' photo of a ' + text[0]  # TODO: rm hardcode
                        if 'textual_inversion' in text_cond:
                            text = 'a dark ' + "<night>" + ' photo of a ' + text[0]
                        if 'controlweak' in text_cond:
                            text = 'a ' + 'foggy' + ' photo of a ' + text[0]
                        if 'controlstrong' in text_cond:
                            text = 'a ' + 'watercolor' + ' painting of a ' + text[0]
                        if 'dreambooth' in text_cond:
                            # text = 'a dashcam photo at night of ' + text[0] + ' in sks style'
                            text = "a dashcam photo of " + text[0] + " at sks night"
                else:
                    text = self.blip_captions[img_id]['captions']
                c = self.sd_model.get_learned_conditioning(text)
                texts.append(text)
                _cs.append(c)
            c = torch.cat(_cs, dim=0)
            conds.append(c)

        # if 'textual_inversion' in text_cond:
        #     texts = []
        #     _cs = []
        #     for img_meta in img_metas:
        #         img_id = img_meta['ori_filename']
        #         text = self.textual_inversion_captions[img_id]['captions']
        #         c = self.sd_model.get_learned_conditioning(text)
        #         texts.append(text)
        #         _cs.append(c)
        #     c = torch.cat(_cs, dim=0)
        #     conds.append(c)

        if 'class_emb' in text_cond:
            c = self.class_embeddings.repeat(bsz, 1, 1)
            # c_crossattn = self.text_adapter(latents, self.class_embeddings,
            #                                 self.gamma)  # NOTE: here the c_crossattn should be expand_dim as latents
            conds.append(c)

        c_crossattn = torch.cat(conds, dim=1)
        if self.opt_dict['use_text_adapter']:
            c_crossattn = self.text_adapter(c_crossattn, self.gamma)

        return c_crossattn

    def extract_feat(self, img, img_metas):
        """Extract features from images."""
        if self.opt_dict.get('use_scaled_encode', False):
            with torch.no_grad():
                latents = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(img))
        else:
            with torch.no_grad():
                latents = self.encoder_vq.encode(img)
            latents = latents.mode().detach()

        c_crossattn = self.create_text_embeddings(img_metas, latents)

        t = torch.ones((img.shape[0],), device=img.device).long()
        outs = self.unet(latents, t, c_crossattn=[c_crossattn])
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

        losses = dict()
        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        return losses

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

    def single_image_inference(self, img, text: str):
        assert self.opt_dict["text_conditioning"] == 'prompt_input', \
            "This method is only available when model is configured for manual prompt input."
        shape = img.shape
        meta = {
            "filename": "none",
            "ori_filename": "none",
            "ori_shape": shape,
            "img_shape": shape,
            "pad_shape": shape,
            "scale_factor": np.asarray([1., 1., 1., 1.]),
            "flip": False,
            "flip_direction": "none",
            "img_norm_cfg": {
                "mean": np.asarray([127.5, 127.5, 127.5]),
                "std": np.asarray([127.5, 127.5, 127.5]),
            },
            "to_rgb": True,
            "prompt": text,
        }

        # longer_side = max(img.size)
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Lambda(lambda x: x * 255.),
                T.Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
            ]
        )
        img_tensor = transform(img).to(self.device)
        img_tensor = img_tensor.unsqueeze(0)

        pred = self.simple_test(img_tensor, [meta])[0]
        return pred
