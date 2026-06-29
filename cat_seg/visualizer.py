import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


class CATSegVisualizer:
    def __init__(self, vis_save_dir: str = "./noise_vis"):
        self.vis_save_dir = vis_save_dir

    def _sanitize_file_stem(self, file_name: str) -> str:
        stem = os.path.splitext(os.path.basename(str(file_name)))[0]
        if not stem:
            stem = "unnamed"
        return stem.replace(os.sep, "_")

    def _sanitize_label(self, label: str) -> str:
        label = str(label).strip()
        if not label:
            return "unnamed"
        safe = []
        for ch in label:
            if ch.isalnum() or ch in ("-", "_"):
                safe.append(ch)
            else:
                safe.append("_")
        return "".join(safe).strip("_") or "unnamed"

    def _to_uint8(self, array: np.ndarray) -> np.ndarray:
        array = np.asarray(array, dtype=np.float32)
        if array.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        min_v = float(array.min())
        max_v = float(array.max())
        if max_v - min_v < 1e-12:
            return np.zeros(array.shape, dtype=np.uint8)
        array = (array - min_v) / (max_v - min_v)
        return (array * 255.0).clip(0, 255).astype(np.uint8)

    def _save_map(self, array: np.ndarray, save_path: str, color_mode: str = "gray") -> None:
        array = np.asarray(array)
        if array.ndim == 1:
            array = array[None, :]
        if color_mode == "jet":
            img = Image.fromarray(self._jet_colormap(array), mode="RGB")
        else:
            img = Image.fromarray(self._to_uint8(array))
        img.save(save_path)

    def _extract_text_maps(
        self, text_noise: Optional[torch.Tensor], batch_size: int
    ) -> List[np.ndarray]:
        if text_noise is None:
            return [None] * batch_size

        tn = text_noise.detach().float().abs().mean(dim=-1)

        if tn.dim() == 0:
            shared = tn.view(1, 1).cpu().numpy()
            return [shared] * batch_size

        if tn.shape[0] == batch_size:
            while tn.dim() > 3:
                tn = tn.mean(dim=1)
            if tn.dim() == 2:
                tn = tn.unsqueeze(1)
            return [tn[i].cpu().numpy() for i in range(batch_size)]

        while tn.dim() > 2:
            tn = tn.mean(dim=0)
        if tn.dim() == 1:
            tn = tn.unsqueeze(0)
        shared = tn.cpu().numpy()
        return [shared] * batch_size

    def save_noise_visuals(
        self,
        files_name: Optional[List[str]],
        text_noise: Optional[torch.Tensor],
        image_noise: Optional[torch.Tensor],
    ) -> None:
        if not files_name:
            return

        text_dir = os.path.join(self.vis_save_dir, "text_noise")
        image_dir = os.path.join(self.vis_save_dir, "image_noise")
        os.makedirs(text_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

        batch_size = len(files_name)
        text_maps = self._extract_text_maps(text_noise, batch_size)

        image_maps = [None] * batch_size
        if image_noise is not None:
            img = image_noise.detach().float().pow(2).mean(dim=1).sqrt()
            image_maps = [img[i].cpu().numpy() for i in range(min(batch_size, img.shape[0]))]
            if len(image_maps) < batch_size and len(image_maps) > 0:
                image_maps.extend([image_maps[-1]] * (batch_size - len(image_maps)))

        for idx, file_name in enumerate(files_name):
            stem = self._sanitize_file_stem(file_name)
            text_map = text_maps[idx] if idx < len(text_maps) else None
            image_map = image_maps[idx] if idx < len(image_maps) else None

            if text_map is not None:
                self._save_map(
                    text_map,
                    os.path.join(text_dir, f"{stem}_text_noise.png"),
                    color_mode="jet",
                )
            if image_map is not None:
                self._save_map(
                    image_map,
                    os.path.join(image_dir, f"{stem}_image_noise.png"),
                    color_mode="jet",
                )

    def _extract_batch_images(self, input_images) -> List[Optional[torch.Tensor]]:
        if input_images is None:
            return []
        if hasattr(input_images, "tensor"):
            input_images = input_images.tensor
        if isinstance(input_images, torch.Tensor):
            if input_images.dim() == 3:
                return [input_images]
            if input_images.dim() == 4:
                return [input_images[i] for i in range(input_images.shape[0])]
            return []
        if isinstance(input_images, (list, tuple)):
            output = []
            for img in input_images:
                if hasattr(img, "tensor"):
                    img = img.tensor
                if isinstance(img, torch.Tensor) and img.dim() == 4 and img.shape[0] == 1:
                    img = img[0]
                output.append(img if isinstance(img, torch.Tensor) else None)
            return output
        return []

    def _to_rgb_uint8(self, image_like: Optional[torch.Tensor]) -> np.ndarray:
        if image_like is None:
            return np.zeros((64, 64, 3), dtype=np.uint8)

        array = image_like.detach().float().cpu().numpy()
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        elif array.ndim == 3 and array.shape[-1] in (1, 3):
            pass
        elif array.ndim == 2:
            array = array[..., None]
        else:
            array = np.squeeze(array)
            if array.ndim == 2:
                array = array[..., None]
            elif array.ndim == 3 and array.shape[0] in (1, 3):
                array = np.transpose(array, (1, 2, 0))

        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        if array.ndim == 2:
            array = array[..., None]
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        elif array.shape[-1] > 3:
            array = array[..., :3]

        max_v = float(array.max()) if array.size > 0 else 0.0
        min_v = float(array.min()) if array.size > 0 else 0.0
        if max_v <= 1.0 and min_v >= 0.0:
            array = array * 255.0
        elif max_v > 255.0 or min_v < 0.0:
            denom = max(max_v - min_v, 1e-12)
            array = (array - min_v) / denom * 255.0

        return array.clip(0, 255).astype(np.uint8)

    def _jet_colormap(
        self,
        array: Optional[np.ndarray],
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> np.ndarray:
        if array is None:
            return np.zeros((64, 64, 3), dtype=np.uint8)

        array = np.asarray(array, dtype=np.float32)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        array = np.squeeze(array)
        if array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.ndim == 1:
            array = array[None, :]
        elif array.ndim > 2:
            array = array.reshape(array.shape[-2], array.shape[-1])

        if vmin is None:
            vmin = float(array.min()) if array.size > 0 else 0.0
        if vmax is None:
            vmax = float(array.max()) if array.size > 0 else 0.0
        if vmax - vmin < 1e-12:
            norm = np.zeros_like(array, dtype=np.float32)
        else:
            norm = (array - vmin) / (vmax - vmin)

        x = np.clip(norm, 0.0, 1.0)
        r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
        rgb = np.stack([r, g, b], axis=-1)
        return (rgb * 255.0).astype(np.uint8)

    def _map_to_pil(
        self,
        array: Optional[np.ndarray],
        size: Optional[Tuple[int, int]] = None,
        panel_type: str = "gray",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> Image.Image:
        if array is None:
            array = np.zeros((64, 64), dtype=np.uint8)
        array = np.asarray(array)
        if array.ndim == 3 and array.shape[-1] == 3:
            img = Image.fromarray(array.astype(np.uint8), mode="RGB")
        else:
            if panel_type == "jet":
                color = self._jet_colormap(array, vmin=vmin, vmax=vmax)
                img = Image.fromarray(color, mode="RGB")
            else:
                gray = self._to_uint8(array)
                if gray.ndim == 3 and gray.shape[-1] == 1:
                    gray = gray[..., 0]
                img = Image.fromarray(gray).convert("RGB")
        if size is not None:
            img = img.resize(size, Image.BILINEAR)
        return img

    def _feature_to_map(self, feature_tensor: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        if feature_tensor is None:
            return None
        feat = feature_tensor.detach().float()
        if feat.dim() == 4:
            feat = feat[0]
        if feat.dim() == 3:
            feat = feat.mean(dim=0)
        elif feat.dim() == 2:
            pass
        else:
            feat = feat.reshape(1, -1)
        return feat.cpu().numpy()

    def _load_vis_base_image(
        self,
        file_name: Optional[str],
        input_image: Optional[torch.Tensor],
    ) -> np.ndarray:
        if file_name is not None and os.path.exists(file_name):
            img = Image.open(file_name).convert("RGB")
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            return img

        rgb = self._to_rgb_uint8(input_image)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _normalize_heatmap(
        self,
        x: np.ndarray,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        eps: float = 1e-6,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if vmin is None:
            vmin = float(x.min()) if x.size > 0 else 0.0
        if vmax is None:
            vmax = float(x.max()) if x.size > 0 else 0.0

        if vmax - vmin < eps:
            return np.zeros_like(x, dtype=np.float32)

        x = (x - vmin) / (vmax - vmin + eps)
        return np.clip(x, 0.0, 1.0)

    def _resize_map_to_image(
        self,
        x: np.ndarray,
        out_h: int,
        out_w: int,
    ) -> np.ndarray:
        if x is None:
            return np.zeros((out_h, out_w), dtype=np.float32)

        tensor = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(
            tensor,
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        )
        return tensor.squeeze(0).squeeze(0).cpu().numpy()

    def _overlay_heatmap_on_bgr(
        self,
        base_bgr: np.ndarray,
        heatmap_2d: np.ndarray,
        alpha: float = 0.6,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> np.ndarray:
        heatmap = self._normalize_heatmap(heatmap_2d, vmin=vmin, vmax=vmax)
        heatmap_uint8 = (heatmap * 255.0).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        vis = cv2.addWeighted(base_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
        return vis

    def _normalize_signed_map(
        self,
        x: np.ndarray,
        clip_percentile: float = 99.0,
        eps: float = 1e-6,
    ) -> Tuple[np.ndarray, float]:
        x = np.asarray(x, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        lim = float(np.percentile(np.abs(x), clip_percentile)) if x.size > 0 else 0.0
        lim = max(lim, eps)
        x = np.clip(x, -lim, lim) / lim
        return x, lim

    def _signed_map_to_bgr(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -1.0, 1.0)
        abs_x = np.abs(x)

        r = np.where(x >= 0, 1.0, 1.0 - abs_x)
        g = 1.0 - abs_x
        b = np.where(x <= 0, 1.0, 1.0 - abs_x)

        rgb = np.stack([r, g, b], axis=-1)
        rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _overlay_signed_heatmap_on_bgr(
        self,
        base_bgr: np.ndarray,
        signed_map_2d: np.ndarray,
        alpha: float = 0.6,
        clip_percentile: float = 99.0,
    ) -> np.ndarray:
        signed_map, _ = self._normalize_signed_map(
            signed_map_2d, clip_percentile=clip_percentile
        )
        signed_color = self._signed_map_to_bgr(signed_map)
        vis = cv2.addWeighted(base_bgr, 1.0 - alpha, signed_color, alpha, 0)
        return vis

    def _reduce_prompt_corr(
        self,
        corr_b: torch.Tensor,
        reduce_type: str = "mean",
    ) -> torch.Tensor:
        if reduce_type == "mean":
            return corr_b.mean(dim=0)
        elif reduce_type == "max":
            return corr_b.max(dim=0).values
        else:
            raise ValueError(f"Unsupported prompt reduce type: {reduce_type}")

    def _make_corr_triptych(
        self,
        base_bgr: np.ndarray,
        before_map: np.ndarray,
        delta_map: Optional[np.ndarray],
        after_map: np.ndarray,
        save_path: str,
        class_name: str,
        score: float,
        alpha: float = 0.6,
    ) -> None:
        H, W = base_bgr.shape[:2]

        before_resized = self._resize_map_to_image(before_map, H, W)
        after_resized = self._resize_map_to_image(after_map, H, W)

        pair_min = float(min(before_resized.min(), after_resized.min()))
        pair_max = float(max(before_resized.max(), after_resized.max()))

        before_vis = self._overlay_heatmap_on_bgr(
            base_bgr, before_resized, alpha=alpha, vmin=pair_min, vmax=pair_max
        )
        after_vis = self._overlay_heatmap_on_bgr(
            base_bgr, after_resized, alpha=alpha, vmin=pair_min, vmax=pair_max
        )

        if delta_map is not None:
            delta_resized = self._resize_map_to_image(delta_map, H, W)
            delta_vis = self._overlay_signed_heatmap_on_bgr(
                base_bgr, delta_resized, alpha=alpha, clip_percentile=99.0
            )
        else:
            delta_vis = base_bgr.copy()

        original_vis = base_bgr.copy()

        title_h = 28
        canvas = np.ones((H + title_h, W * 4, 3), dtype=np.uint8) * 255
        panels = [original_vis, before_vis, delta_vis, after_vis]
        titles = ["image", "corr_before", "delta_corr", "corr_after"]

        for i, (panel, title) in enumerate(zip(panels, titles)):
            x0 = i * W
            canvas[title_h:title_h + H, x0:x0 + W] = panel
            cv2.putText(
                canvas,
                title,
                (x0 + 8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        footer = f"class={class_name} | score={score:.4f}"
        cv2.putText(
            canvas,
            footer,
            (8, H + title_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            footer,
            (8, H + title_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.imwrite(save_path, canvas)

    def save_corr_visuals(
        self,
        files_name: Optional[List[str]],
        class_names: List[str],
        input_images,
        corr_before: Optional[torch.Tensor],
        delta_corr: Optional[torch.Tensor],
        corr_after: Optional[torch.Tensor],
        prompt_reduce: str = "mean",
        topk: int = 8,
        score_reduce: str = "mean",
        alpha: float = 0.6,
    ) -> None:
        if not files_name or corr_before is None or corr_after is None:
            return

        batch_images = self._extract_batch_images(input_images)
        b_lim = min(len(files_name), corr_before.shape[0], corr_after.shape[0])

        for b_idx in range(b_lim):
            file_name = files_name[b_idx]
            stem = self._sanitize_file_stem(file_name)
            save_dir = os.path.join(self.vis_save_dir, "corr_overlay", stem)
            os.makedirs(save_dir, exist_ok=True)

            input_image = batch_images[b_idx] if b_idx < len(batch_images) else None
            base_bgr = self._load_vis_base_image(file_name, input_image)

            corr_before_bt = self._reduce_prompt_corr(corr_before[b_idx], reduce_type=prompt_reduce)
            corr_after_bt = self._reduce_prompt_corr(corr_after[b_idx], reduce_type=prompt_reduce)

            delta_corr_bt = None
            if delta_corr is not None and b_idx < delta_corr.shape[0]:
                delta_corr_bt = self._reduce_prompt_corr(delta_corr[b_idx], reduce_type=prompt_reduce)

            if score_reduce == "mean":
                cls_scores = corr_after_bt.flatten(1).mean(dim=1)
            elif score_reduce == "max":
                cls_scores = corr_after_bt.flatten(1).max(dim=1).values
            else:
                raise ValueError(f"Unsupported score_reduce: {score_reduce}")

            k = min(int(topk), corr_after_bt.shape[0])
            topk_idx = torch.topk(cls_scores, k=k, dim=0).indices.tolist()

            for rank, t_idx in enumerate(topk_idx):
                class_name = class_names[t_idx] if t_idx < len(class_names) else f"class_{t_idx}"
                class_tag = self._sanitize_label(class_name)

                before_map = corr_before_bt[t_idx].detach().float().cpu().numpy()
                after_map = corr_after_bt[t_idx].detach().float().cpu().numpy()

                delta_map = None
                if delta_corr_bt is not None and t_idx < delta_corr_bt.shape[0]:
                    delta_map = delta_corr_bt[t_idx].detach().float().cpu().numpy()

                score = float(cls_scores[t_idx].item())
                save_path = os.path.join(
                    save_dir,
                    f"{rank:02d}_t{t_idx:03d}_{class_tag}_score{score:.4f}.png",
                )

                self._make_corr_triptych(
                    base_bgr=base_bgr,
                    before_map=before_map,
                    delta_map=delta_map,
                    after_map=after_map,
                    save_path=save_path,
                    class_name=class_name,
                    score=score,
                    alpha=alpha,
                )


    def save_corr_visuals_by_gt(
        self,
        files_name: Optional[List[str]],
        class_names: List[str],
        input_images,
        corr_before: Optional[torch.Tensor],
        delta_corr: Optional[torch.Tensor],
        corr_after: Optional[torch.Tensor],
        targets: Optional[torch.Tensor],
        ignore_value: int = 255,
        prompt_reduce: str = "mean",
        alpha: float = 0.6,
    ) -> None:
        if not files_name or corr_before is None or corr_after is None or targets is None:
            return

        batch_images = self._extract_batch_images(input_images)
        b_lim = min(
            len(files_name),
            corr_before.shape[0],
            corr_after.shape[0],
            targets.shape[0],
        )

        for b_idx in range(b_lim):
            file_name = files_name[b_idx]
            stem = self._sanitize_file_stem(file_name)

            gt_save_dir = os.path.join(self.vis_save_dir, "gt_corr_overlay", stem)
            non_gt_save_dir = os.path.join(self.vis_save_dir, "non_gt_corr_overlay", stem)
            os.makedirs(gt_save_dir, exist_ok=True)
            os.makedirs(non_gt_save_dir, exist_ok=True)

            input_image = batch_images[b_idx] if b_idx < len(batch_images) else None
            base_bgr = self._load_vis_base_image(file_name, input_image)

            corr_before_bt = self._reduce_prompt_corr(
                corr_before[b_idx], reduce_type=prompt_reduce
            )
            corr_after_bt = self._reduce_prompt_corr(
                corr_after[b_idx], reduce_type=prompt_reduce
            )

            delta_corr_bt = None
            if delta_corr is not None and b_idx < delta_corr.shape[0]:
                delta_corr_bt = self._reduce_prompt_corr(
                    delta_corr[b_idx], reduce_type=prompt_reduce
                )

            target_b = targets[b_idx]
            valid = target_b != ignore_value
            if not valid.any():
                continue

            gt_classes = torch.unique(target_b[valid]).tolist()
            gt_classes = [
                int(c) for c in gt_classes
                if 0 <= int(c) < corr_after_bt.shape[0]
            ]

            for cls_idx in gt_classes:
                class_name = (
                    class_names[cls_idx]
                    if cls_idx < len(class_names)
                    else f"class_{cls_idx}"
                )
                class_tag = self._sanitize_label(class_name)

                gt_mask = ((target_b == cls_idx) & valid).detach().cpu().numpy().astype(np.uint8)
                non_gt_mask = ((target_b != cls_idx) & valid).detach().cpu().numpy().astype(np.uint8)

                before_map = corr_before_bt[cls_idx].detach().float().cpu().numpy()
                after_map = corr_after_bt[cls_idx].detach().float().cpu().numpy()

                delta_map = None
                if delta_corr_bt is not None and cls_idx < delta_corr_bt.shape[0]:
                    delta_map = delta_corr_bt[cls_idx].detach().float().cpu().numpy()

                score = float(corr_after_bt[cls_idx].flatten().mean().item())

                gt_save_path = os.path.join(
                    gt_save_dir,
                    f"t{cls_idx:03d}_{class_tag}.png",
                )
                non_gt_save_path = os.path.join(
                    non_gt_save_dir,
                    f"t{cls_idx:03d}_{class_tag}.png",
                )

                self._make_corr_five_panel(
                    base_bgr=base_bgr,
                    gt_mask=gt_mask,
                    before_map=before_map,
                    delta_map=delta_map,
                    after_map=after_map,
                    save_path=gt_save_path,
                    class_name=class_name,
                    score=score,
                    alpha=alpha,
                    mask_title="gt_mask",
                )

                self._make_corr_five_panel(
                    base_bgr=base_bgr,
                    gt_mask=non_gt_mask,
                    before_map=before_map,
                    delta_map=delta_map,
                    after_map=after_map,
                    save_path=non_gt_save_path,
                    class_name=class_name,
                    score=score,
                    alpha=alpha,
                    mask_title="non_gt_mask",
                )
                
    def _binary_mask_to_bgr(self, mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.ndim != 2:
            mask = np.squeeze(mask)
        mask = (mask > 0).astype(np.uint8) * 255
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    def _make_corr_five_panel(
        self,
        base_bgr: np.ndarray,
        gt_mask: np.ndarray,
        before_map: np.ndarray,
        delta_map: Optional[np.ndarray],
        after_map: np.ndarray,
        save_path: str,
        class_name: str,
        score: float,
        alpha: float = 0.6,
        mask_title: str = "mask",
    ) -> None:
        H, W = base_bgr.shape[:2]

        gt_mask_resized = self._resize_map_to_image(gt_mask.astype(np.float32), H, W)
        gt_mask_vis = self._binary_mask_to_bgr(gt_mask_resized > 0.5)

        before_resized = self._resize_map_to_image(before_map, H, W)
        after_resized = self._resize_map_to_image(after_map, H, W)

        pair_min = float(min(before_resized.min(), after_resized.min()))
        pair_max = float(max(before_resized.max(), after_resized.max()))

        before_vis = self._overlay_heatmap_on_bgr(
            base_bgr, before_resized, alpha=alpha, vmin=pair_min, vmax=pair_max
        )
        after_vis = self._overlay_heatmap_on_bgr(
            base_bgr, after_resized, alpha=alpha, vmin=pair_min, vmax=pair_max
        )

        if delta_map is not None:
            delta_resized = self._resize_map_to_image(delta_map, H, W)
            delta_vis = self._overlay_signed_heatmap_on_bgr(
                base_bgr, delta_resized, alpha=alpha, clip_percentile=99.0
            )
        else:
            delta_vis = base_bgr.copy()

        original_vis = base_bgr.copy()

        title_h = 28
        canvas = np.ones((H + title_h, W * 5, 3), dtype=np.uint8) * 255
        panels = [original_vis, gt_mask_vis, before_vis, delta_vis, after_vis]
        titles = ["image", mask_title, "corr_before", "delta_corr", "corr_after"]

        for i, (panel, title) in enumerate(zip(panels, titles)):
            x0 = i * W
            canvas[title_h:title_h + H, x0:x0 + W] = panel
            cv2.putText(
                canvas,
                title,
                (x0 + 8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        footer = f"class={class_name} | score={score:.4f}"
        cv2.putText(
            canvas,
            footer,
            (8, H + title_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            footer,
            (8, H + title_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.imwrite(save_path, canvas)
        
    def _build_four_panel_visual(
        self,
        file_name: str,
        input_image: Optional[torch.Tensor],
        image_feature: Optional[torch.Tensor],
        image_noise: Optional[torch.Tensor],
        noisy_image_feature: Optional[torch.Tensor],
    ) -> None:
        grid_dir = os.path.join(self.vis_save_dir, "feature_noise_grid")
        os.makedirs(grid_dir, exist_ok=True)

        original_img = self._map_to_pil(self._to_rgb_uint8(input_image), panel_type="image")
        target_size = original_img.size
        if target_size[0] <= 1 or target_size[1] <= 1:
            target_size = (224, 224)
            original_img = original_img.resize(target_size, Image.BILINEAR)

        feat_img = self._map_to_pil(
            self._feature_to_map(image_feature), size=target_size, panel_type="jet"
        )

        noise_map = None
        if image_noise is not None:
            noise_map = image_noise.detach().float().pow(2).mean(dim=0).sqrt().cpu().numpy()
        noise_img = self._map_to_pil(noise_map, size=target_size, panel_type="jet")

        noisy_feat_img = self._map_to_pil(
            self._feature_to_map(noisy_image_feature), size=target_size, panel_type="jet"
        )

        panels = [original_img, feat_img, noise_img, noisy_feat_img]
        titles = ["image", "image_feature", "noise", "image_feature_plus_noise"]

        title_h = 24
        canvas = Image.new("RGB", (target_size[0] * 4, target_size[1] + title_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        for idx, (panel, title) in enumerate(zip(panels, titles)):
            x0 = idx * target_size[0]
            canvas.paste(panel, (x0, title_h))
            draw.text((x0 + 6, 4), title, fill=(0, 0, 0))

        stem = self._sanitize_file_stem(file_name)
        canvas.save(os.path.join(grid_dir, f"{stem}_grid.png"))

    def save_visual_grid(
        self,
        files_name: Optional[List[str]],
        input_images,
        image_feature: torch.Tensor,
        image_noise: Optional[torch.Tensor],
        noisy_image_feature: torch.Tensor,
    ) -> None:
        if not files_name:
            return

        batch_images = self._extract_batch_images(input_images)
        batch_size = len(files_name)
        for idx, file_name in enumerate(files_name):
            input_image = batch_images[idx] if idx < len(batch_images) else None
            feat = image_feature[idx] if image_feature is not None and idx < image_feature.shape[0] else None
            noise = image_noise[idx] if image_noise is not None and idx < image_noise.shape[0] else None
            noisy_feat = (
                noisy_image_feature[idx]
                if noisy_image_feature is not None and idx < noisy_image_feature.shape[0]
                else None
            )
            self._build_four_panel_visual(file_name, input_image, feat, noise, noisy_feat)