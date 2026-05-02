from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from collections import OrderedDict
from typing import Literal


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import logging
from romav2.device import device
from romav2.features import Descriptor, FineFeatures
from romav2.geometry import (
    bhwc_grid_sample,
    bhwc_interpolate,
    get_normalized_grid,
    prec_mat_from_prec_params,
    to_pixel,
)
from romav2.io import check_not_i16
from romav2.matcher import Matcher
from romav2.refiner import Refiners
from romav2.types import Setting, ImageLike

logger = logging.getLogger(__name__)

_WEIGHT_URLS = {
    "v2.0.0": "https://github.com/Parskatt/RoMaV2/releases/download/weights/romav2.pt",
    "v2.0.1": "https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt",
}


def _interpolate_warp_and_confidence(
    *,
    warp: torch.Tensor,
    confidence: torch.Tensor,
    H: int,
    W: int,
    patch_size: int,
    zero_out_precision: bool,
):
    warp = bhwc_interpolate(
        warp.detach(),
        size=(H // patch_size, W // patch_size),
        mode="bilinear",
        align_corners=False,
    )
    if zero_out_precision:
        # delta at 4 is absolute, and if we
        # for the second pass we therefore can't use first pred.
        # overlap is fine since it's relative to matcher pred.
        confidence[..., 1:] = 0.0

    confidence = bhwc_interpolate(
        confidence.detach(),
        size=(H // patch_size, W // patch_size),
        mode="bilinear",
        align_corners=False,
    )
    return warp, confidence


def _map_confidence(*, confidence: torch.Tensor, threshold: float | None):
    overlap = confidence[..., :1].sigmoid()
    if threshold is not None:
        overlap[overlap > threshold] = 1.0
    precision = prec_mat_from_prec_params(confidence[..., 1:4])
    return overlap, precision


class RoMaV2(nn.Module):
    @dataclass(frozen=True)
    class Cfg:
        descriptor: Descriptor.Cfg = Descriptor.Cfg()
        matcher: Matcher.Cfg = Matcher.Cfg()
        refiners: Refiners.Cfg = Refiners.Cfg()
        refiner_features: FineFeatures.Cfg = FineFeatures.Cfg()
        anchor_width: int = 512
        anchor_height: int = 512
        setting: Setting = "precise"
        compile: bool = True
        name: str = "RoMa v2"
        use_true_highres: bool = False
        version: Literal["v2.0.0", "v2.0.1"] = "v2.0.0"
        weights_url: str | None = None
        force_native_local_corr: bool = False

    # settings
    H_lr: int
    W_lr: int
    H_hr: int | None
    W_hr: int | None
    bidirectional: bool
    threshold: float | None
    balanced_sampling: bool

    def __init__(self, cfg: Cfg | None = None):
        super().__init__()
        if cfg is None:
            # default
            cfg = RoMaV2.Cfg()
        if cfg.version not in _WEIGHT_URLS:
            raise ValueError(f"Unknown RoMaV2 version: {cfg.version}")

        cfg = replace(
            cfg,
            refiners=replace(
                cfg.refiners,
                local_corr_mode=cfg.version,
                force_native_local_corr=cfg.force_native_local_corr,
            ),
        )
        weights_url = cfg.weights_url or _WEIGHT_URLS[cfg.version]
        if cfg.version == "v2.0.0" and cfg.weights_url is None:
            weights = torch.hub.load_state_dict_from_url(weights_url)
        else:
            weights = torch.hub.load_state_dict_from_url(weights_url, map_location=device)
        self.f = Descriptor(cfg.descriptor)
        self.matcher = Matcher(cfg.matcher)
        self.cfg = cfg
        self.anchor_width = cfg.anchor_width
        self.anchor_height = cfg.anchor_height
        self.refiners = Refiners(cfg.refiners)
        self.refiner_features = FineFeatures(cfg.refiner_features)
        self.to(device)
        self.eval()
        self.apply_setting(cfg.setting)
        self.name = cfg.name
        self.load_state_dict(weights)
        if cfg.compile:
            logger.info(f"Compiling {self.name}...")
            self.compile()
        logger.info(f"{self.name} initialized.")

    def apply_setting(self, setting: Setting):
        if setting in ["mega1500", "scannet1500", "wxbs", "satast"]:
            self.H_lr = 800
            self.W_lr = 800
            self.H_hr = 1024
            self.W_hr = 1024
            self.bidirectional = True
            self.threshold = 0.05
            self.balanced_sampling = True
        elif setting == "turbo":
            self.H_lr = 320
            self.W_lr = 320
            self.H_hr = None
            self.W_hr = None
            self.bidirectional = False
            self.threshold = None
            self.balanced_sampling = True
        elif setting == "fast":
            self.H_lr = 512
            self.W_lr = 512
            self.H_hr = None
            self.W_hr = None
            self.bidirectional = False
            self.threshold = None
            self.balanced_sampling = True
        elif setting == "base":
            self.H_lr = 640
            self.W_lr = 640
            self.H_hr = None
            self.W_hr = None
            self.bidirectional = False
            self.threshold = None
            self.balanced_sampling = True
        elif setting == "precise":
            self.H_lr = 800
            self.W_lr = 800
            self.H_hr = 1280
            self.W_hr = 1280
            self.bidirectional = True
            self.threshold = None
            self.balanced_sampling = True
        else:
            raise TypeError(f"Invalid setting: {setting}")

    @torch.inference_mode()
    def forward(
        self,
        img_A_lr: torch.Tensor,
        img_B_lr: torch.Tensor,
        img_A_hr: torch.Tensor | None = None,
        img_B_hr: torch.Tensor | None = None,
        # NEW ARGUMENT: matches your old 'batch["corresps"]' logic
        previous_preds: dict | None = None,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        if torch.get_float32_matmul_precision() != "highest":
            raise RuntimeError("Float32 matmul precision must be set to highest")
        assert not self.training, "Currently only inference mode released"

        # init preds
        predictions = OrderedDict()

        # --- LOGIC CHANGE START ---
        # If we don't have previous predictions (checkpoint), run the Coarse Matcher (DINO)
        if previous_preds is None:
            f_A = self.f(img_A_lr)
            f_B = self.f(img_B_lr)
            matcher_output = self.matcher(
                f_A,
                f_B,
                img_A=img_A_lr,
                img_B=img_B_lr,
                bidirectional=self.bidirectional,
            )
            predictions["matcher"] = matcher_output

            # Initialize warp/confidence from the coarse matcher
            warp_AB, confidence_AB = (
                matcher_output["warp_AB"],
                matcher_output["confidence_AB"],
            )
            if self.bidirectional:
                warp_BA, confidence_BA = (
                    matcher_output["warp_BA"],
                    matcher_output["confidence_BA"],
                )
            else:
                warp_BA, confidence_BA = None, None

        # If we DO have previous predictions, load them (resume state)
        else:
            # We assume previous_preds contains the state of the flow/warp
            warp_AB = previous_preds["warp_AB"]
            confidence_AB = previous_preds["confidence_AB"]
            warp_BA = previous_preds.get("warp_BA")
            confidence_BA = previous_preds.get("confidence_BA")

            # Carry over all previous info into current predictions
            for k, v in previous_preds.items():
                predictions[k] = v
        # --- LOGIC CHANGE END ---

        # Coarse-only mode: skip refiner, return coarse matcher output directly
        if getattr(self, "coarse_only", False):
            predictions["warp_AB"] = warp_AB
            predictions["confidence_AB"] = confidence_AB
            if self.bidirectional:
                predictions["warp_BA"] = warp_BA
                predictions["confidence_BA"] = confidence_BA
            else:
                predictions["warp_BA"] = None
                predictions["confidence_BA"] = None
            return predictions

        # Refine warp (VGG / Refiner stage)
        # This continues from wherever warp_AB/confidence_AB are currently pointing
        for stage, (img_A, img_B) in enumerate(
            zip([img_A_lr, img_A_hr], [img_B_lr, img_B_hr])
        ):
            if img_A is None or img_B is None:
                continue
            B, C, H, W = img_A.shape
            scale_factor = torch.tensor(
                (W / self.anchor_width, H / self.anchor_height), device=device
            )
            refiner_features_A = self.refiner_features(img_A)
            refiner_features_B = self.refiner_features(img_B)
            for patch_size_str, refiner in self.refiners.items():
                patch_size = int(patch_size_str)
                zero_out_precision = (
                    img_A_hr is not None and patch_size == 4 and stage == 1
                )
                warp_AB, confidence_AB = _interpolate_warp_and_confidence(
                    warp=warp_AB,
                    confidence=confidence_AB,
                    H=H,
                    W=W,
                    patch_size=patch_size,
                    zero_out_precision=zero_out_precision,
                )
                if self.bidirectional:
                    warp_BA, confidence_BA = _interpolate_warp_and_confidence(
                        warp=warp_BA,
                        confidence=confidence_BA,
                        H=H,
                        W=W,
                        patch_size=patch_size,
                        zero_out_precision=zero_out_precision,
                    )

                f_patch_A = refiner_features_A[patch_size]
                f_patch_B = refiner_features_B[patch_size]
                refiner_output_AB = refiner(
                    f_A=f_patch_A,
                    f_B=f_patch_B,
                    prev_warp=warp_AB,
                    prev_confidence=confidence_AB,
                    scale_factor=scale_factor,
                )
                if self.bidirectional:
                    refiner_output_BA = refiner(
                        f_A=f_patch_B,
                        f_B=f_patch_A,
                        prev_warp=warp_BA,
                        prev_confidence=confidence_BA,
                        scale_factor=scale_factor,
                    )
                else:
                    refiner_output_BA = None
                predictions[f"refiner_{patch_size}_AB"] = refiner_output_AB
                predictions[f"refiner_{patch_size}_BA"] = refiner_output_BA
                warp_AB, confidence_AB = (
                    refiner_output_AB["warp"],
                    refiner_output_AB["confidence"],
                )
                if self.bidirectional:
                    warp_BA, confidence_BA = (
                        refiner_output_BA["warp"],
                        refiner_output_BA["confidence"],
                    )
            predictions["warp_AB"] = warp_AB
            predictions["confidence_AB"] = confidence_AB
            if self.bidirectional:
                predictions["warp_BA"] = warp_BA
                predictions["confidence_BA"] = confidence_BA
            else:
                predictions["warp_BA"] = None
                predictions["confidence_BA"] = None
        return predictions

    def _load_image(self, img_like: ImageLike) -> torch.Tensor:
        if isinstance(img_like, str) or isinstance(img_like, Path):
            img_pil = Image.open(img_like)
            check_not_i16(img_pil)
            img_pil = img_pil.convert("RGB")
            img = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).to(device)
        elif isinstance(img_like, Image.Image):
            img = torch.from_numpy(np.array(img_like)).permute(2, 0, 1).to(device)
        elif isinstance(img_like, np.ndarray):
            assert img_like.shape[-1] == 3, (
                f"Image must have 3 channels, but got shape {img_like.shape=}"
            )
            img = torch.from_numpy(img_like).permute(2, 0, 1).to(device)
        elif isinstance(img_like, torch.Tensor):
            assert img_like.shape[1] == 3, (
                f"Image must have 3 channels, but got shape {img_like.shape=}"
            )
            img = img_like
        else:
            raise ValueError(f"Unsupported image type: {type(img_like)}")

        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        if len(img.shape) == 3:
            img = img[None]
        return img

    @torch.inference_mode()
    def match(
        self,
        img_like_A: ImageLike = None,  # Made optional
        img_like_B: ImageLike = None,  # Made optional
        refine_batch: dict | None = None,  # Your "Checkpoint" argument
        im_A_high_res: ImageLike = None,
        im_B_high_res: ImageLike = None,
        **kwargs,
    ) -> tuple[dict[str, torch.Tensor], dict]:  # Returns (preds, cache)
        if self.training:
            self.eval()

        # --- LOGIC CHANGE START ---
        # 1. RESUME MODE
        if refine_batch is not None:
            img_A = refine_batch.get("im_A", refine_batch.get("img_A"))
            img_B = refine_batch.get("im_B", refine_batch.get("img_B"))
            if "corresps" in refine_batch:
                legacy_preds = refine_batch["corresps"]

                # TRANSFORM: Legacy (B,2,H,W) -> RoMaV2 (B,H,W,2)
                warp_AB = legacy_preds["flow"]  # .permute(0, 2, 3, 1)

                # INFLATE: Legacy Probability -> RoMaV2 4-Channel Confidence
                # 1. Permute to (B, H, W, 1)


                confidence_AB = legacy_preds["certainty"]  # .permute(0, 2, 3, 1)
                # # 2. Inverse Sigmoid to get Logits (RoMa v2 expects logits in channel 0)
                # logits = torch.logit(probs.clamp(1e-6, 1 - 1e-6))
                # # 3. Add zero-initialized precision channels (Channels 1-3)
                # B, H, W, _ = logits.shape
                # zeros = torch.zeros(
                #     B, H, W, 3, device=logits.device, dtype=logits.dtype
                # )
                # confidence_AB = torch.cat([logits, zeros], dim=-1)

                previous_preds = {
                    "warp_AB": warp_AB,
                    "confidence_AB": confidence_AB,
                    "warp_BA": None,
                    "confidence_BA": None,
                }
            else:
                # Fallback if you pass the raw RoMaV2 dictionary directly
                previous_preds = refine_batch.get("preds")

        # 2. FRESH START
        else:
            if img_like_A is None or img_like_B is None:
                raise ValueError("If refine_batch is None, images must be provided.")

            img_A = self._load_image(img_like_A)
            img_B = self._load_image(img_like_B)
            previous_preds = None
        # --- LOGIC CHANGE END ---

        # Prepare LR inputs
        img_A_lr = F.interpolate(
            img_A,
            size=(self.H_lr, self.W_lr),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        img_B_lr = F.interpolate(
            img_B,
            size=(self.H_lr, self.W_lr),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

        # Prepare HR inputs (if settings allow)
        if self.H_hr is not None and self.W_hr is not None:
            if self.cfg.use_true_highres and im_A_high_res is not None:
                img_A_hr = F.interpolate(
                    self._load_image(im_A_high_res),
                    size=(self.H_hr, self.W_hr),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
            else:
                img_A_hr = F.interpolate(
                    img_A,
                    size=(self.H_hr, self.W_hr),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
            if self.cfg.use_true_highres and im_B_high_res is not None:
                img_B_hr = F.interpolate(
                    self._load_image(im_B_high_res),
                    size=(self.H_hr, self.W_hr),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
            else:
                img_B_hr = F.interpolate(
                    img_B,
                    size=(self.H_hr, self.W_hr),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
        else:
            img_A_hr = None
            img_B_hr = None

        # Call forward, injecting previous_preds if we are resuming
        preds = self(
            img_A_lr,
            img_B_lr,
            img_A_hr=img_A_hr,
            img_B_hr=img_B_hr,
            previous_preds=previous_preds,  # Pass it down
        )

        # Standard RoMaV2 post-processing
        warp_AB = preds["warp_AB"]
        confidence_AB = preds["confidence_AB"]
        warp_BA = preds["warp_BA"]
        confidence_BA = preds["confidence_BA"]
        if getattr(self, "coarse_only", False):
            overlap_AB = confidence_AB[..., :1].sigmoid()
            precision_AB = torch.zeros(*overlap_AB.shape[:-1], 2, 2, device=overlap_AB.device)
            if self.bidirectional and confidence_BA is not None:
                overlap_BA = confidence_BA[..., :1].sigmoid()
                precision_BA = torch.zeros(*overlap_BA.shape[:-1], 2, 2, device=overlap_BA.device)
            else:
                overlap_BA = None
                precision_BA = None
        else:
            overlap_AB, precision_AB = _map_confidence(
                confidence=confidence_AB, threshold=self.threshold
            )
            if self.bidirectional:
                overlap_BA, precision_BA = _map_confidence(
                    confidence=confidence_BA, threshold=self.threshold
                )
            else:
                overlap_BA = None
                precision_BA = None

        final_preds = {
            "warp_AB": warp_AB.clone(),
            "confidence_AB": confidence_AB.clone(),
            "overlap_AB": overlap_AB.clone(),
            "precision_AB": precision_AB.clone(),
            "warp_BA": warp_BA.clone() if warp_BA is not None else None,
            "confidence_BA": (
                confidence_BA.clone() if confidence_BA is not None else None
            ),
            "overlap_BA": overlap_BA.clone() if overlap_BA is not None else None,
            "precision_BA": precision_BA.clone() if precision_BA is not None else None,
        }

        # --- CACHE CREATION ---
        # Store everything needed to resume later in 'refine_batch' style
        # Skip cache for coarse_only mode (keyframing doesn't need it)
        if getattr(self, "coarse_only", False):
            cache = {}
        else:
            cache = {
                "batch": {
                    "img_A": img_A,
                    "img_B": img_B,
                    "corresps": final_preds,
                }
            }

        # Return tuple (results, cache) to match your workflow
        return final_preds, cache

    def sample(self, preds: dict[str, torch.Tensor], num_corresp: int):
        warp = preds["warp_AB"]
        confidence_AB = preds["overlap_AB"]
        precision_AB = preds["precision_AB"] if "precision_AB" in preds else None

        warp = warp[0]
        confidence_AB = confidence_AB[0].reshape(-1)
        if precision_AB is not None:
            precision_AB = precision_AB[0]

        H_A, W_A, two = warp.shape
        grid = get_normalized_grid(1, H_A, W_A)[0]
        matches_AB = torch.cat((grid, warp), dim=-1).reshape(-1, 4)
        if self.bidirectional:
            confidence_BA = preds["overlap_BA"]
            warp_BA = preds["warp_BA"]

            precision_BA = (
                preds["precision_BA"] if "precision_BA" in preds else None
            )
            warp_BA = warp_BA[0]
            confidence_BA = confidence_BA[0]
            if precision_BA is not None:
                precision_BA = precision_BA[0]

            if precision_BA is not None and precision_AB is not None:
                precision_A = bhwc_grid_sample(
                    precision_BA[None].reshape(1, H_A, W_A, -1),
                    warp[None],
                    mode="bilinear",
                    align_corners=False,
                ).reshape(H_A, W_A, 2, 2)
                precision_B = bhwc_grid_sample(
                    precision_AB[None].reshape(1, H_A, W_A, -1),
                    warp_BA[None],
                    mode="bilinear",
                    align_corners=False,
                ).reshape(H_A, W_A, 2, 2)
                precision_fwd = torch.stack(
                    (precision_A, precision_AB), dim=-3
                ).reshape(-1, 2, 2, 2)
                precision_bwd = torch.stack(
                    (precision_BA, precision_B), dim=-3
                ).reshape(-1, 2, 2, 2)
                precision = torch.cat((precision_fwd, precision_bwd), dim=0)
            else:
                precision = None
            # let's hope H_A is equal to H_B
            grid = get_normalized_grid(1, H_A, W_A)[0]
            matches_BA = torch.cat((warp_BA, grid), dim=-1).reshape(-1, 4)
            confidence = torch.cat(
                (confidence_AB.reshape(-1), confidence_BA.reshape(-1)), dim=0
            )
            matches = torch.cat((matches_AB, matches_BA), dim=0)
        else:
            matches = matches_AB
            confidence = confidence_AB.reshape(-1)
            precision = precision_AB.reshape(-1, 2, 2)

        expansion_factor = 4
        confidence = confidence * matches.abs().amax(dim=-1).le(1 - 1 / H_A).float()
        corresp_inds = torch.multinomial(
            confidence, expansion_factor * num_corresp, replacement=False
        )
        sampled_matches = matches[corresp_inds]
        sampled_confidence = confidence[corresp_inds]
        if precision is not None:
            sampled_precision = precision[corresp_inds]
        else:
            sampled_precision = None
        # return sampled_matches, sampled_confidence
        density = kde(sampled_matches)

        p = 1 / (density + 1)
        p[density < 10] = (
            1e-7  # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        )
        balanced_samples = torch.multinomial(
            p, num_samples=min(num_corresp, len(sampled_confidence)), replacement=False
        )
        return (
            sampled_matches[balanced_samples],
            sampled_confidence[balanced_samples],
            sampled_precision[balanced_samples][:, 0]
            if sampled_precision is not None
            else None,
            sampled_precision[balanced_samples][:, 1]
            if sampled_precision is not None
            else None,
        )

    @classmethod
    def prec_map_coordinates(
        cls, precision: torch.Tensor, *, H_in: int, W_in: int, H_out: int, W_out: int
    ):
        W_ratio = W_in / W_out
        H_ratio = H_in / H_out
        ratio = torch.tensor([W_ratio, H_ratio], device=precision.device)
        precision = precision * ratio[None, :, None] * ratio[None, None, :]
        return precision

    @classmethod
    def to_pixel_coordinates(
        cls, warp: torch.Tensor, H_A: int, W_A: int, H_B: int, W_B: int
    ):
        return to_pixel(warp[..., :2], H=H_A, W=W_A), to_pixel(
            warp[..., 2:], H=H_B, W=W_B
        )

    @classmethod
    def match_keypoints(
        cls,
        x_A: torch.Tensor,
        x_B: torch.Tensor,
        warp: torch.Tensor,
        certainty: torch.Tensor,
        return_tuple: bool = True,
        return_inds: bool = False,
        max_dist: float = 0.005,
        cert_th: float = 0.0,
    ):
        x_A_to_B = F.grid_sample(
            warp[..., -2:].permute(2, 0, 1)[None],
            x_A[None, None],
            align_corners=False,
            mode="bilinear",
        )[0, :, 0].mT
        cert_A_to_B = F.grid_sample(
            certainty[None, None, ...],
            x_A[None, None],
            align_corners=False,
            mode="bilinear",
        )[0, 0, 0]
        D = torch.cdist(x_A_to_B, x_B)
        inds_A, inds_B = torch.nonzero(
            (D == D.min(dim=-1, keepdim=True).values)
            * (D == D.min(dim=-2, keepdim=True).values)
            * (cert_A_to_B[:, None] > cert_th)
            * (D < max_dist),
            as_tuple=True,
        )

        if return_tuple:
            if return_inds:
                return inds_A, inds_B
            else:
                return x_A[inds_A], x_B[inds_B]
        else:
            if return_inds:
                return torch.cat((inds_A, inds_B), dim=-1)
            else:
                return torch.cat((x_A[inds_A], x_B[inds_B]), dim=-1)


def kde(x: torch.Tensor, std: float = 0.1, half: bool = True) -> torch.Tensor:
    # use a gaussian kernel to estimate density
    if half:
        x = x.half()
    scores = (-(torch.cdist(x, x) ** 2) / (2 * std**2)).exp()
    density = scores.sum(dim=-1)
    return density
