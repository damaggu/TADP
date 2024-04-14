"""
Mostly adapted from VPD.
TODO link to VPD file
"""

from typing import List
import torch
import torch.nn as nn
import torchvision.transforms as T
from timm.models.layers import trunc_normal_, DropPath
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import torch.nn.functional as F
import json
from TADP.vpd.models import UNetWrapper, TextAdapterDepth


class TADPDepthEncoder(nn.Module):
    def __init__(self, out_dim=1024, ldm_prior=[320, 640, 1280 + 1280], sd_path=None, text_dim=768,
                 dataset='nyu', opt_dict=None,
                 ):
        super().__init__()

        self.opt_dict = opt_dict
        self.layer1 = nn.Sequential(
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
            nn.GroupNorm(16, ldm_prior[0]),
            nn.ReLU(),
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(ldm_prior[1], ldm_prior[1], 3, stride=2, padding=1),
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(ldm_prior), out_dim, 1),
            nn.GroupNorm(16, out_dim),
            nn.ReLU(),
        )

        self.apply(self._init_weights)

        ### stable diffusion layers

        config = OmegaConf.load('models/depth/configs/v1-inference.yaml')
        if sd_path is None:
            config.model.params.ckpt_path = 'checkpoints/v1-5-pruned-emaonly.ckpt'
        else:
            config.model.params.ckpt_path = f'{sd_path}'
            # config.model.params.ckpt_path = f'../{sd_path}'

        sd_model = instantiate_from_config(config.model)
        self.encoder_vq = sd_model.first_stage_model

        self.unet = UNetWrapper(sd_model.model, use_attn=False)

        _tc = self.opt_dict["text_conditioning"]
        if not ("blip" in _tc or _tc == "prompt_input"):
            del sd_model.cond_stage_model
        elif "blip" in _tc:
            with open(self.opt_dict['blip_caption_path'], 'r') as f:
                print('Loaded blip captions!')
                self.blip_captions = json.load(f)
                # get max length
                self.blip_max_length = max([len(caption) for caption in self.blip_captions])
        del self.encoder_vq.decoder
        del self.unet.unet.diffusion_model.out
        self.sd_model = sd_model

        for param in self.encoder_vq.parameters():
            param.requires_grad = False

        if dataset == 'nyu':
            self.text_adapter = TextAdapterDepth(text_dim=text_dim)
            class_embeddings = torch.load('TADP/vpd/nyu_class_embeddings.pth')
        else:
            raise NotImplementedError

        if not self.opt_dict['use_text_adapter']:
            self.text_adapter = None

        self.register_buffer('class_embeddings', class_embeddings)  # TODO don't we need this only in VPD setting?
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, feats):
        x = self.ldm_to_net[0](feats[0])
        for i in range(3):
            if i > 0:
                x = x + self.ldm_to_net[i](feats[i])
            x = self.layers[i](x)
            x = self.upsample_layers[i](x)
        return self.out_conv(x)

    def create_text_embeddings(self, latents, img_metas, class_ids=None):
        bsz = latents.shape[0]

        conds = []
        texts = []
        _cs = []

        if self.opt_dict['text_conditioning'] == "prompt_input":
            captions = img_metas.get("prompts")
            assert captions is not None
            for text in captions:
                c = self.sd_model.get_learned_conditioning([text])
                texts.append(text)
                _cs.append(c)
            c = torch.cat(_cs, dim=0)
            conds.append(c)
        else:
            text_cond = self.opt_dict['text_conditioning'].split('+')

            if 'blip' in text_cond:
                for img_id in img_metas['img_paths']:
                    text = self.blip_captions[img_id]['captions']
                    c = self.sd_model.get_learned_conditioning(text)
                    texts.append(text)
                    _cs.append(c)
                c = torch.cat(_cs, dim=0)
                conds.append(c)

            if 'class_emb' in text_cond:
                if class_ids is not None:
                    class_embeddings = self.class_embeddings[class_ids.tolist()]
                else:
                    class_embeddings = self.class_embeddings

                c = class_embeddings.repeat(bsz, 1, 1)
                # c_crossattn = self.text_adapter(latents, class_embeddings,
                #                                 self.gamma)  # NOTE: here the c_crossattn should be expand_dim as latents
                conds.append(c)

        c_crossattn = torch.cat(conds, dim=1)

        if self.opt_dict['use_text_adapter']:
            c_crossattn = self.text_adapter(c_crossattn, self.gamma)
        return c_crossattn

    def forward(self, x, img_metas, class_ids=None):
        if self.opt_dict.get('use_scaled_encode', False):
            with torch.no_grad():
                latents = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(x))
        else:
            with torch.no_grad():
                latents = self.encoder_vq.encode(x)
            latents = latents.mode().detach()

        c_crossattn = self.create_text_embeddings(latents, img_metas, class_ids)
        t = torch.ones((x.shape[0],), device=x.device).long()
        # import pdb; pdb.set_trace()
        outs = self.unet(latents, t, c_crossattn=[c_crossattn])
        feats = [outs[0], outs[1], torch.cat([outs[2], F.interpolate(outs[3], scale_factor=2)], dim=1)]
        x = torch.cat([self.layer1(feats[0]), self.layer2(feats[1]), feats[2]], dim=1)
        return self.out_layer(x)


class TADPDepth(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.max_depth = args.max_depth
        self.args = args

        embed_dim = 192

        channels_in = embed_dim * 8
        channels_out = embed_dim

        if args.dataset == 'nyudepthv2':
            self.encoder = TADPDepthEncoder(out_dim=channels_in, dataset='nyu', sd_path=args.sd_ckpt_path,
                                           opt_dict=args.__dict__)
        else:
            raise NotImplementedError

        self.decoder = Decoder(channels_in, channels_out, args)
        self.decoder.init_weights()

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))

        for m in self.last_layer_depth.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)


    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, metas, class_ids=None):
        # import pdb; pdb.set_trace()
        b, c, h, w = x.shape
        x = x * 2.0 - 1.0  # normalize to [-1, 1]
        if h == 480 and w == 480:
            new_x = torch.zeros(b, c, 512, 512, device=x.device)
            new_x[:, :, 0:480, 0:480] = x
            x = new_x
        elif h == 352 and w == 352:
            new_x = torch.zeros(b, c, 384, 384, device=x.device)
            new_x[:, :, 0:352, 0:352] = x
            x = new_x
        elif h == 512 and w == 512:
            pass
        else:
            raise NotImplementedError
        conv_feats = self.encoder(x, metas, class_ids)

        if h == 480 or h == 352:
            conv_feats = conv_feats[:, :, :-1, :-1]

        out = self.decoder([conv_feats])
        out_depth = self.last_layer_depth(out)
        out_depth = torch.sigmoid(out_depth) * self.max_depth

        return {'pred_d': out_depth}

    @torch.no_grad()
    def test_inference(self, input_RGB: torch.tensor, prompts: List[str]):
        if self.args.shift_window_test:
            bs, _, h, w = input_RGB.shape
            assert w > h and bs == 1
            interval_all = w - h
            interval = interval_all // (self.args.shift_size - 1)
            sliding_images = []
            sliding_masks = torch.zeros((bs, 1, h, w), device=input_RGB.device)
            for i in range(self.args.shift_size):
                sliding_images.append(input_RGB[..., :, i * interval:i * interval + h])
                sliding_masks[..., :, i * interval:i * interval + h] += 1
            input_RGB = torch.cat(sliding_images, dim=0)
        if self.args.flip_test:
            input_RGB = torch.cat((input_RGB, torch.flip(input_RGB, [3])), dim=0)
        if input_RGB.shape[0] > 1:
            prompts = prompts * input_RGB.shape[0]
        meta = {
            "img_paths": None,
            "prompts": prompts
        }
        pred = self.forward(input_RGB, meta)
        pred_d = pred['pred_d']

        if self.args.flip_test:
            batch_s = pred_d.shape[0] // 2
            pred_d = (pred_d[:batch_s] + torch.flip(pred_d[batch_s:], [3])) / 2.0

        if self.args.shift_window_test:
            pred_s = torch.zeros((bs, 1, h, w), device=pred_d.device)
            for i in range(self.args.shift_size):
                pred_s[..., :, i * interval:i * interval + h] += pred_d[i:i + 1]
            pred_d = pred_s / sliding_masks
        pred_d = pred_d * 1000.0  # for nyu

        return pred_d

    def single_image_inference(self, img, text: str):
        assert self.encoder.opt_dict["text_conditioning"] == 'prompt_input', \
            "This method is only available when model is configured for manual prompt input."
        transform = T.ToTensor()
        img = transform(img).to(self.device)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        pred = self.test_inference(img, [text])[0].permute(1, 2, 0).cpu().numpy()
        return pred


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        self.deconv = args.num_deconv
        self.in_channels = in_channels

        # import pdb; pdb.set_trace()

        self.deconv_layers = self._make_deconv_layer(
            args.num_deconv,
            args.num_filters,
            args.deconv_kernels,
        )

        conv_layers = []
        conv_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=args.num_filters[-1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1))
        conv_layers.append(
            build_norm_layer(dict(type='BN'), out_channels)[1])
        conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*conv_layers)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, conv_feats):
        # import pdb; pdb.set_trace()
        out = self.deconv_layers(conv_feats[0])
        out = self.conv_layers(out)

        out = self.up(out)
        out = self.up(out)

        return out

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""

        layers = []
        in_planes = self.in_channels
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
