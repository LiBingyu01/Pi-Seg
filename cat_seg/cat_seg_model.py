# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple
import os

import csv
import json
import torch.distributed as dist

import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from einops import rearrange
    
@META_ARCH_REGISTRY.register()
class CATSeg(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        sliding_window: bool,
        clip_finetune: str,
        backbone_multiplier: float,
        clip_pretrained: str,
        output_dir: str,
    ):
        """
        Args:
            sem_seg_head: a module that predicts semantic segmentation from backbone features
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json
        self.output_dir = output_dir
        
        self.clip_finetune = clip_finetune
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
            if "transformer" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "attention":
                    if "attn" in name:
                        # QV fine-tuning for attention blocks
                        params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                    elif "position" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False

        self.sliding_window = sliding_window
        self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)
        
        self.proj_dim = 768 if clip_pretrained == "ViT-B/16" else 1024
        self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4)

        self.layer_indexes = [3, 7] if clip_pretrained == "ViT-B/16" else [7, 15] 
        self.layers = []
        for l in self.layer_indexes:
            self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks[l].register_forward_hook(lambda m, _, o: self.layers.append(o))
        
        # delta corr stats print control
        self.delta_corr_print_freq = 20
        self._delta_corr_step = 0
        self.latest_delta_corr_stats = None

        # delta corr log paths -> under OUTPUT_DIR
        self.delta_corr_log_dir = os.path.join(self.output_dir, "delta_corr_logs")
        self.delta_corr_csv_path = os.path.join(self.delta_corr_log_dir, "delta_corr_step_log.csv")
        self.delta_corr_jsonl_path = os.path.join(self.delta_corr_log_dir, "delta_corr_step_log.jsonl")
        if self.training:
            self._init_delta_corr_logger()

    @classmethod
    def from_config(cls, cfg):
        backbone = None
        sem_seg_head = build_sem_seg_head(cfg, None)
        
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
            "output_dir": cfg.OUTPUT_DIR,
        }

    @property
    def device(self):
        return self.pixel_mean.device
    def _is_dist_avail_and_initialized(self):
        return dist.is_available() and dist.is_initialized()

    def _is_main_process(self):
        return (not self._is_dist_avail_and_initialized()) or dist.get_rank() == 0

    def _sync_tensor_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self._is_dist_avail_and_initialized():
            return tensor
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def _init_delta_corr_logger(self):
        """
        Clear old logs at startup, then recreate fresh CSV header.
        Only rank 0 does file ops; others wait.
        """
        if self._is_main_process():
            os.makedirs(self.delta_corr_log_dir, exist_ok=True)

            if os.path.exists(self.delta_corr_csv_path):
                os.remove(self.delta_corr_csv_path)
            if os.path.exists(self.delta_corr_jsonl_path):
                os.remove(self.delta_corr_jsonl_path)

            with open(self.delta_corr_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step",
                    "gt_in_mean",
                    "non_gt_mean",
                    "gap",
                    "align_ratio",
                ])

        if self._is_dist_avail_and_initialized():
            dist.barrier()

    def _append_delta_corr_log(self, gt_in_mean: float, non_gt_mean: float):
        gap = gt_in_mean - non_gt_mean
        align_ratio = gt_in_mean / (abs(non_gt_mean) + 1e-6)

        with open(self.delta_corr_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self._delta_corr_step,
                gt_in_mean,
                non_gt_mean,
                gap,
                align_ratio,
            ])

        with open(self.delta_corr_jsonl_path, "a") as f:
            record = {
                "step": int(self._delta_corr_step),
                "gt_in_mean": gt_in_mean,
                "non_gt_mean": non_gt_mean,
                "gap": gap,
                "align_ratio": align_ratio,
            }
            f.write(json.dumps(record) + "\n")

    def _compute_delta_corr_stats(self, delta_corr, targets, ignore_value):
        """
        delta_corr: (B, P, T, Hc, Wc)
        targets:    (B, H, W), class index mask

        Returns:
            dict with local sums/counts + local means
        """
        if delta_corr is None:
            return None

        # prompt mean: (B, P, T, Hc, Wc) -> (B, T, Hc, Wc)
        delta_corr = delta_corr.mean(dim=1)

        # resize to GT mask size
        delta_corr = F.interpolate(
            delta_corr,
            size=targets.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        valid = targets != ignore_value

        gt_in_sum = 0.0
        gt_in_cnt = 0.0
        non_gt_sum = 0.0
        non_gt_cnt = 0.0

        B, T, H, W = delta_corr.shape
        for b in range(B):
            valid_b = valid[b]
            if not valid_b.any():
                continue

            present_classes = torch.unique(targets[b][valid_b])

            for cls in present_classes:
                cls = int(cls.item())
                if cls < 0 or cls >= T:
                    continue

                pos_mask = (targets[b] == cls) & valid_b
                neg_mask = (targets[b] != cls) & valid_b

                if pos_mask.any():
                    pos_vals = delta_corr[b, cls][pos_mask]
                    gt_in_sum += float(pos_vals.sum().item())
                    gt_in_cnt += float(pos_vals.numel())

                if neg_mask.any():
                    neg_vals = delta_corr[b, cls][neg_mask]
                    non_gt_sum += float(neg_vals.sum().item())
                    non_gt_cnt += float(neg_vals.numel())

        gt_in_mean = gt_in_sum / max(gt_in_cnt, 1.0)
        non_gt_mean = non_gt_sum / max(non_gt_cnt, 1.0)

        return {
            "gt_in_sum": gt_in_sum,
            "gt_in_cnt": gt_in_cnt,
            "non_gt_sum": non_gt_sum,
            "non_gt_cnt": non_gt_cnt,
            "gt_in_mean": gt_in_mean,
            "non_gt_mean": non_gt_mean,
        }

    def _sync_delta_corr_stats(self, local_stats):
        """
        Sum sums/counts across all ranks, then compute true global mean.
        """
        if local_stats is None:
            return None

        stats_tensor = torch.tensor(
            [
                local_stats["gt_in_sum"],
                local_stats["gt_in_cnt"],
                local_stats["non_gt_sum"],
                local_stats["non_gt_cnt"],
            ],
            device=self.device,
            dtype=torch.float64,
        )

        stats_tensor = self._sync_tensor_sum(stats_tensor)

        gt_in_sum = float(stats_tensor[0].item())
        gt_in_cnt = float(stats_tensor[1].item())
        non_gt_sum = float(stats_tensor[2].item())
        non_gt_cnt = float(stats_tensor[3].item())

        gt_in_mean = gt_in_sum / max(gt_in_cnt, 1.0)
        non_gt_mean = non_gt_sum / max(non_gt_cnt, 1.0)
        gap = gt_in_mean - non_gt_mean
        align_ratio = gt_in_mean / (abs(non_gt_mean) + 1e-6)

        return {
            "gt_in_sum": gt_in_sum,
            "gt_in_cnt": gt_in_cnt,
            "non_gt_sum": non_gt_sum,
            "non_gt_cnt": non_gt_cnt,
            "gt_in_mean": gt_in_mean,
            "non_gt_mean": non_gt_mean,
            "gap": gap,
            "align_ratio": align_ratio,
        }    
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        
        images = [x["image"].to(self.device) for x in batched_inputs]
        if not self.training and self.sliding_window:
            return self.inference_sliding_window(batched_inputs)

        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)

        self.layers = []

        clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized, dense=True)
        image_features = clip_features[:, 1:, :]

        # CLIP ViT features for guidance
        res3 = rearrange(image_features, "B (H W) C -> B C H W", H=24)
        res4 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24)
        res5 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24)
        res4 = self.upsample1(res4)
        res5 = self.upsample2(res5)
        features = {'res5': res5, 'res4': res4, 'res3': res3,}

        # files_name = [x["file_name"] for x in batched_inputs]
        input_images = [x["image"] for x in batched_inputs]
        # targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
        outputs = self.sem_seg_head(
            clip_features,
            features,
            # files_name=files_name,
            files_name=None,
            input_images=input_images,
            targets=None,
        )
        if self.training:
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            outputs = F.interpolate(outputs, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)

            # ============================================================ #
            # 1) local stats on each rank
            local_delta_corr_stats = self._compute_delta_corr_stats(
                self.sem_seg_head.predictor.cached_delta_corr,
                targets,
                self.sem_seg_head.ignore_value,
            )

            # 2) global multi-GPU average
            global_delta_corr_stats = self._sync_delta_corr_stats(local_delta_corr_stats)
            self.latest_delta_corr_stats = global_delta_corr_stats

            self._delta_corr_step += 1

            # 3) only rank 0 logs and prints
            if self._is_main_process() and global_delta_corr_stats is not None:
                self._append_delta_corr_log(
                    global_delta_corr_stats["gt_in_mean"],
                    global_delta_corr_stats["non_gt_mean"],
                )

                if (
                    self.delta_corr_print_freq > 0
                    and self._delta_corr_step % self.delta_corr_print_freq == 0
                ):
                    print(
                        "[DELTA_CORR] "
                        f"step={self._delta_corr_step}, "
                        f"gt_in_mean={global_delta_corr_stats['gt_in_mean']:.6f}, "
                        f"non_gt_mean={global_delta_corr_stats['non_gt_mean']:.6f}, "
                        f"gap={global_delta_corr_stats['gap']:.6f}, "
                        f"align_ratio={global_delta_corr_stats['align_ratio']:.6f}"
                    )
            # ============================================================ #
            
            num_classes = outputs.shape[1]
            mask = targets != self.sem_seg_head.ignore_value

            outputs = outputs.permute(0,2,3,1)
            _targets = torch.zeros(outputs.shape, device=self.device)
            _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
            _targets[mask] = _onehot
            
            loss = F.binary_cross_entropy_with_logits(outputs, _targets)
            losses = {"loss_sem_seg" : loss}
            return losses

        else:
            outputs = outputs.sigmoid()
            image_size = clip_images.image_sizes[0]
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])

            output = sem_seg_postprocess(outputs[0], image_size, height, width)
            processed_results = [{'sem_seg': output}]
            return processed_results


    @torch.no_grad()
    def inference_sliding_window(self, batched_inputs, kernel=384, overlap=0.333, out_res=[640, 640]):
        images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
        stride = int(kernel * (1 - overlap))
        unfold = nn.Unfold(kernel_size=kernel, stride=stride)
        fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

        image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False).squeeze()
        image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
        global_image = F.interpolate(images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear', align_corners=False)
        image = torch.cat((image, global_image), dim=0)

        images = (image - self.pixel_mean) / self.pixel_std
        clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
        clip_images = F.interpolate(clip_images, size=self.clip_resolution, mode='bilinear', align_corners=False, )
        
        self.layers = []
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True)
        res3 = rearrange(clip_features[:, 1:, :], "B (H W) C -> B C H W", H=24)
        res4 = self.upsample1(rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24))
        res5 = self.upsample2(rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24))

        features = {'res5': res5, 'res4': res4, 'res3': res3,}
        outputs = self.sem_seg_head(clip_features, features)

        outputs = F.interpolate(outputs, size=kernel, mode="bilinear", align_corners=False)
        outputs = outputs.sigmoid()
        
        global_output = outputs[-1:]
        global_output = F.interpolate(global_output, size=out_res, mode='bilinear', align_corners=False,)
        outputs = outputs[:-1]
        outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1] + out_res, device=self.device)))
        outputs = (outputs + global_output) / 2.

        height = batched_inputs[0].get("height", out_res[0])
        width = batched_inputs[0].get("width", out_res[1])
        output = sem_seg_postprocess(outputs[0], out_res, height, width)
        return [{'sem_seg': output}]
