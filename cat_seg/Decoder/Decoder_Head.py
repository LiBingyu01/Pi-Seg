# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified by Jian Ding from: https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
# Modified by Heeseong Shin from: https://github.com/dingjiansw101/ZegFormer/blob/main/mask_former/mask_former_model.py

import logging
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import fvcore.nn.weight_init as weight_init
import open_clip
import os

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from .Cost_Aggregation import Aggregator
from cat_seg.clip import clip
from cat_seg.clip import imagenet_templates
from cat_seg.pini_vpn import ImageVpnGenerator, TextVpnGenerator
from cat_seg.visualizer import CATSegVisualizer


class CATSegPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        train_class_json: str,
        test_class_json: str,
        clip_pretrained: str,
        prompt_ensemble_type: str,
        text_guidance_dim: int,
        text_guidance_proj_dim: int,
        appearance_guidance_dim: int,
        appearance_guidance_proj_dim: int,
        prompt_depth: int,
        prompt_length: int,
        decoder_dims: list,
        decoder_guidance_dims: list,
        decoder_guidance_proj_dims: list,
        num_heads: int,
        num_layers: tuple,
        hidden_dims: tuple,
        pooling_sizes: tuple,
        feature_resolution: tuple,
        window_sizes: tuple,
        attention_type: str,
        pini_enabled: bool = True,
        pini_image_vpn_enabled: bool = True,
        pini_text_vpn_enabled: bool = True,
        pini_reduction: int = 1,
        pini_text_noise_std: float = 0.02,

        # ===== image noise =====
        pini_image_noise_type: str = "laplace",
        pini_image_student_t_df: float = 3.0,

        # ===== text noise =====
        pini_text_noise_type: str = "uniform",
        pini_text_student_t_df: float = 3.0,

        is_vis: bool = False,
        vis_save_dir: str = "./delta_corr_vis",
    ):
        """
        Args:
            
        """
        super().__init__()
        
        self.is_vis = is_vis
        self.visualizer = CATSegVisualizer(vis_save_dir)
        
        import json
        # use class_texts in train_forward, and test_class_texts in test_forward
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts
        device = "cuda" if torch.cuda.is_available() else "cpu"
  
        self.tokenizer = None
        if clip_pretrained == "ViT-G" or clip_pretrained == "ViT-H":
            # for OpenCLIP models
            name, pretrain = ('ViT-H-14', 'laion2b_s32b_b79k') if clip_pretrained == 'ViT-H' else ('ViT-bigG-14', 'laion2b_s39b_b160k')
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                name, 
                pretrained=pretrain, 
                device=device, 
                force_image_size=336,)
        
            self.tokenizer = open_clip.get_tokenizer(name)
        else:
            # for OpenAI models
            clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)
    
        self.prompt_ensemble_type = prompt_ensemble_type        

        if self.prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            prompt_templates = ['A photo of a {} in the scene',]
        else:
            raise NotImplementedError
        
        self.prompt_templates = prompt_templates

        self.text_features = self.class_embeddings(self.class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        self.text_features_test = self.class_embeddings(self.test_class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        
        self.clip_model = clip_model.float()
        self.clip_preprocess = clip_preprocess
        
        transformer = Aggregator(
            text_guidance_dim=text_guidance_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            appearance_guidance_dim=appearance_guidance_dim,
            appearance_guidance_proj_dim=appearance_guidance_proj_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            num_layers=num_layers,
            nheads=num_heads,
            hidden_dim=hidden_dims,
            pooling_size=pooling_sizes,
            feature_resolution=feature_resolution,
            window_size=window_sizes,
            attention_type=attention_type,
            prompt_channel=len(prompt_templates),
            )
        self.transformer = transformer

        # PiNI: feature-level noise generators
        self.image_vpn = None
        self.text_vpn = None
        if pini_enabled:
            if pini_image_vpn_enabled:
                self.image_vpn = ImageVpnGenerator(
                    clip_dim=text_guidance_dim,
                    reduction=pini_reduction,
                    noise_type=pini_image_noise_type,
                    student_t_df=pini_image_student_t_df,
                )
            if pini_text_vpn_enabled:
                self.text_vpn = TextVpnGenerator(
                    clip_dim=text_guidance_dim,
                    noise_std=pini_text_noise_std,
                    noise_type=pini_text_noise_type,
                    student_t_df=pini_text_student_t_df,
                )

        self.tokens = None
        self.cache = None
        self.cached_corr_before = None
        self.cached_corr_after = None
        self.cached_delta_corr = None
    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1)
        text_feats = F.normalize(text_feats, dim=-1)
        corr = torch.einsum("bchw, btpc -> bpthw", img_feats, text_feats)
        return corr    

    @classmethod
    def from_config(cls, cfg):#, in_channels, mask_classification):
        ret = {}

        ret["train_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON
        ret["test_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON
        ret["clip_pretrained"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
        ret["prompt_ensemble_type"] = cfg.MODEL.PROMPT_ENSEMBLE_TYPE

        # Aggregator parameters:
        ret["text_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM
        ret["text_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM
        ret["appearance_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM
        ret["appearance_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM

        ret["decoder_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS
        ret["decoder_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS
        ret["decoder_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS

        ret["prompt_depth"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH
        ret["prompt_length"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH

        ret["num_layers"] = cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS
        ret["num_heads"] = cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS
        ret["hidden_dims"] = cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS
        ret["pooling_sizes"] = cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES
        ret["feature_resolution"] = cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION
        ret["window_sizes"] = cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES
        ret["attention_type"] = cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE

        # PiNI config
        ret["pini_enabled"] = cfg.MODEL.PINI.ENABLED
        ret["pini_image_vpn_enabled"] = cfg.MODEL.PINI.IMAGE_VPN_ENABLED
        ret["pini_text_vpn_enabled"] = cfg.MODEL.PINI.TEXT_VPN_ENABLED
        ret["pini_reduction"] = cfg.MODEL.PINI.REDUCTION
        ret["pini_text_noise_std"] = cfg.MODEL.PINI.TEXT_NOISE_STD

        # ===== image noise =====
        ret["pini_image_noise_type"] = cfg.MODEL.PINI.IMAGE_NOISE_TYPE
        ret["pini_image_student_t_df"] = cfg.MODEL.PINI.IMAGE_STUDENT_T_DF

        # ===== text noise =====
        ret["pini_text_noise_type"] = cfg.MODEL.PINI.TEXT_NOISE_TYPE
        ret["pini_text_student_t_df"] = cfg.MODEL.PINI.TEXT_STUDENT_T_DF

        ret["is_vis"] = cfg.MODEL.PINI.IS_VIS
        ret["vis_save_dir"] = os.path.join(cfg.OUTPUT_DIR, "vis_noise")

        return ret

    def forward(self, 
                x, 
                vis_guidance, 
                prompt=None, 
                gt_cls=None,
                files_name=None,
                input_images=None,
                targets=None,
                ):
        vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]
        class_names = self.class_texts if self.training else self.test_class_texts
        class_names = [class_names[c] for c in gt_cls] if gt_cls is not None else class_names
        text = self.get_text_embeds(class_names, self.prompt_templates, self.clip_model, prompt)

        # PiNI: text feature noise (training only)
        if (self.training or self.is_vis) and self.text_vpn is not None:
            text = self.text_vpn(text)


        text = text.repeat(x.shape[0], 1, 1, 1)
        image_feature_before_noise = x.detach()
        # PiNI: image feature noise guided by text cross-attention (training only)
        if (self.training or self.is_vis) and self.image_vpn is not None:
            B, C, H, W = x.shape
            spatial_feat = rearrange(x, 'B C H W -> B (H W) C')
            # Mean-pool text across prompts: (B,T,P,C) -> (T,C)
            text_for_attn = text[0].mean(dim=1).detach()
            mu, variance = self.image_vpn(spatial_feat.detach(), text_for_attn)
            im_noise = self.image_vpn.sample(mu, variance)
            x = x + rearrange(im_noise, 'B (H W) C -> B C H W', H=H, W=W)
        image_feature_after_noise = x.detach()
        
        
        corr_before = None
        corr_after = None
        delta_corr = None
        if self.training or self.is_vis:
            corr_before = self.correlation(image_feature_before_noise, text)
            corr_after = self.correlation(image_feature_after_noise, text)
            delta_corr = corr_after - corr_before
            self.cached_corr_before = corr_before.detach()
            self.cached_corr_after = corr_after.detach()
            self.cached_delta_corr = delta_corr.detach()
        else:
            self.cached_corr_before = None
            self.cached_corr_after = None
            self.cached_delta_corr = None
        
        if self.is_vis:
            self.visualizer.save_noise_visuals(files_name, None, im_noise)
            self.visualizer.save_visual_grid(
                files_name=files_name,
                input_images=input_images,
                image_feature=image_feature_before_noise,
                image_noise=im_noise,
                noisy_image_feature=image_feature_after_noise,
            )
            
            # ==== GT
            self.visualizer.save_corr_visuals_by_gt(
                files_name=files_name,
                class_names=class_names,
                input_images=input_images,
                corr_before=corr_before,
                delta_corr=delta_corr,
                corr_after=corr_after,
                targets=targets,
                ignore_value=255,
                prompt_reduce="mean",
                alpha=0.60,
            )
            # ===== TOPK
            # self.visualizer.save_corr_visuals(
            #     files_name=files_name,
            #     class_names=class_names,
            #     input_images=input_images,
            #     corr_before=corr_before,
            #     delta_corr=delta_corr,
            #     corr_after=corr_after,
            #     targets=targets,
            #     ignore_value=255,
            #     prompt_reduce="mean",
            #     topk=8,
            #     score_reduce="mean",
            #     alpha=0.60,
            #     use_gt_only=True,
            #     gt_select_mode="largest",
            # )

            if im_noise is not None:
                noise_ratio = (
                    im_noise.detach().norm(dim=1).mean()
                    / (image_feature_before_noise.detach().norm(dim=1).mean() + 1e-6)
                ).item()
                delta_ratio = (
                    delta_corr.detach().abs().mean()
                    / (corr_before.detach().abs().mean() + 1e-6)
                ).item()
                print(f"[VIS] noise_ratio={noise_ratio:.6f}, delta_ratio={delta_ratio:.6f}")


        out = self.transformer(x, text, vis)
        return out

    @torch.no_grad()
    def class_embeddings(self, classnames, templates, clip_model):
        zeroshot_weights = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname) for template in templates]  # format with class
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).cuda()
            else: 
                texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights
    
    def get_text_embeds(self, classnames, templates, clip_model, prompt=None):
        if self.cache is not None and not self.training:
            return self.cache
        
        if self.tokens is None or prompt is not None:
            tokens = []
            for classname in classnames:
                if ', ' in classname:
                    classname_splits = classname.split(', ')
                    texts = [template.format(classname_splits[0]) for template in templates]
                else:
                    texts = [template.format(classname) for template in templates]  # format with class
                if self.tokenizer is not None:
                    texts = self.tokenizer(texts).cuda()
                else: 
                    texts = clip.tokenize(texts).cuda()
                tokens.append(texts)
            tokens = torch.stack(tokens, dim=0).squeeze(1)
            if prompt is None:
                self.tokens = tokens
        elif self.tokens is not None and prompt is None:
            tokens = self.tokens

        class_embeddings = clip_model.encode_text(tokens, prompt)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        
        class_embeddings = class_embeddings.unsqueeze(1)
        
        if not self.training:
            self.cache = class_embeddings
            
        return class_embeddings


# ============================================================
# CATSeg Head (Detectron2)
# ============================================================

@SEM_SEG_HEADS_REGISTRY.register()
class CATSegHead(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int,
        feature_resolution: list,
        transformer_predictor: nn.Module,
    ):
        super().__init__()
        self.ignore_value = ignore_value
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution
        self.predictor = transformer_predictor

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return dict(
            ignore_value=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            feature_resolution=cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            transformer_predictor=CATSegPredictor(cfg),
        )

    def forward(self, 
                features, 
                guidance_features, 
                prompt=None, 
                gt_cls=None, 
                files_name=None, 
                input_images=None,
                targets=None,
                ):
        img_feat = rearrange(
            features[:, 1:, :],
            "b (h w) c -> b c h w",
            h=self.feature_resolution[0],
            w=self.feature_resolution[1],
        )
        return self.predictor(
            img_feat,
            guidance_features,
            prompt,
            gt_cls,
            files_name=files_name,
            input_images=input_images,
            targets=targets,
        )