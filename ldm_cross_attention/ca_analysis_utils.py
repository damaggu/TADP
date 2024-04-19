import torch
import numpy as np
from torch import nn
import json

class TextConditioningWrapper:


    def __init__(self, cfg, sd_model, class_embedding_path='./data/pascal_class_embeddings.pth'):
        self.cfg = cfg
        self.cfg['original_tc_str'] = self.cfg['text_conditioning']
        self.texts = []

        # turn text conditioning into list
        if '+' in self.cfg['text_conditioning']:
            self.cfg['text_conditioning'] = self.cfg['text_conditioning'].split('+')
        else:
            self.cfg['text_conditioning'] = [self.cfg['text_conditioning']]

        eos_token = 49407
        self.eos_token = eos_token
        if 'class_names' in self.cfg['text_conditioning']:
            self.class_names = self.cfg['class_names']
            with torch.no_grad():
                sd_model.cond_stage_model.to('cuda')
                class_emb_stack = []
                all_tokens = None
                all_pos = 0

                for i, class_name in enumerate(self.class_names):
                    _emb, tokens = sd_model.get_learned_conditioning(class_name, return_tokens=True)
                    if all_tokens is None:
                        eos_pos = torch.where(tokens == eos_token)[1][0].item()
                        all_tokens = tokens
                        all_pos = eos_pos
                        class_emb_stack.append(_emb[:, :eos_pos])
                    else:
                        eos_pos = torch.where(tokens == eos_token)[1][0].item()
                        all_tokens[:, all_pos:(all_pos + eos_pos - 1)] = tokens[:, 1:eos_pos]
                        all_pos += (eos_pos - 1)
                        class_emb_stack.append(_emb[:, 1:eos_pos])

                self.class_names_embs = torch.cat(class_emb_stack, dim=1)
                num_pad = 77 - self.class_names_embs.size(1)
                self.class_names_embs = torch.cat([self.class_names_embs, _emb[[0], None, -1].repeat(1, num_pad, 1)], dim=1)
                self.class_names_tokens = all_tokens

        if 'blip' in self.cfg["original_tc_str"]:
            import os
            os.getcwd()
            os.path.isfile(self.cfg['caption_path'])
            with open(self.cfg['caption_path'], 'r') as f:
                self.blip_captions = json.load(f)
                # get max length
                self.blip_max_length = max([len(caption) for caption in self.blip_captions])

        self.sd_model = sd_model
        self.class_embeddings = torch.load(class_embedding_path).to('cuda')
        self.text_dim = self.class_embeddings.size(-1)
        self.n_classes = self.class_embeddings.size(0)

    def create_text_embeddings(self, img_metas):
        bsz = len(img_metas['img_id'])
        text_cond = self.cfg['text_conditioning']
        conds = []
        texts = None
        tokens_list = []
        if 'blip' in text_cond:
            texts = []
            _cs = []
            _tokens = []
            for img_id in img_metas['img_id']:
                text = self.blip_captions[img_id]['captions']
                c, tokens = self.sd_model.get_learned_conditioning(text, return_tokens=True)
                texts.append(text)
                _cs.append(c)
                _tokens.append(tokens)
            c = torch.cat(_cs, dim=0)
            conds.append(c)
            tokens = torch.cat(_tokens, dim=0)
            tokens_list.append(tokens)

        if 'shuffled_blip' in text_cond:
            texts = []
            _cs = []
            _tokens = []
            for img_id in img_metas['img_id']:
                text = self.blip_captions[img_id]['captions']
                # split text by words and shuffle
                text = [word for caption in text for word in caption.split(' ')]
                np.random.shuffle(text)
                text = [' '.join(text)]
                c, tokens = self.sd_model.get_learned_conditioning(text, return_tokens=True)
                texts.append(text)
                _tokens.append(tokens)
                _cs.append(c)
            c = torch.cat(_cs, dim=0)
            conds.append(c)
            tokens = torch.cat(_tokens, dim=0)
            tokens_list.append(tokens)

        if 'generic_str' in text_cond:
            texts = []
            _cs = []
            _tokens = []

            for img_id in img_metas['img_id']:
                text = ['a photo of something.']
                c, tokens = self.sd_model.get_learned_conditioning(text, return_tokens=True)
                texts.append(text)
                _cs.append(c)
                _tokens.append(tokens)
            c = torch.cat(_cs, dim=0)
            conds.append(c)
            tokens = torch.cat(_tokens, dim=0)
            tokens_list.append(tokens)

        if 'blank_str' in text_cond:
            texts = []
            _cs = []
            _tokens = []

            for img_id in img_metas['img_id']:
                text = ['']
                c, tokens = self.sd_model.get_learned_conditioning(text, return_tokens=True)
                texts.append(text)
                _cs.append(c)
                _tokens.append(tokens)

            c = torch.cat(_cs, dim=0)
            conds.append(c)
            tokens = torch.cat(_tokens, dim=0)
            tokens_list.append(tokens)

        if 'class_names' in text_cond:
            _cs = []
            _tokens = []

            for img_id in img_metas['img_id']:
                _cs.append(self.class_names_embs)
                _tokens.append(self.class_names_tokens)
            c = torch.cat(_cs, dim=0)
            conds.append(c)
            tokens = torch.cat(_tokens, dim=0)
            tokens_list.append(tokens)

        if 'class_emb' in text_cond:
            c_crossattn = self.class_embeddings.repeat(bsz, 1, 1)
            conds.append(c_crossattn)
            tokens_list = []
            tmp = torch.ones(size=(2, self.class_embeddings.shape[0] + 1,
                                   self.class_embeddings.shape[1]), dtype=torch.int64).to('cuda')
            tmp[:, -1, :] = self.eos_token
            tokens_list.append(tmp)

        if "custom_prompt" in text_cond:
            texts = []
            _cs = []
            _tokens = []

            for img_id in img_metas['img_id']:
                text = [self.cfg["custom_prompt"]]
                c, tokens = self.sd_model.get_learned_conditioning(text, return_tokens=True)
                texts.append(text)
                _cs.append(c)
                _tokens.append(tokens)

            c = torch.cat(_cs, dim=0)
            conds.append(c)
            tokens = torch.cat(_tokens, dim=0)
            tokens_list.append(tokens)

        c_crossattn = torch.cat(conds, dim=1)
        tokens = torch.cat(tokens_list, dim=1)
        # if self.cfg['use_text_adapter']:
        #     c_crossattn = self.text_adapter(c_crossattn,
        #                                     self.gamma)

        if texts is not None:
            self.texts = texts

        return c_crossattn, tokens