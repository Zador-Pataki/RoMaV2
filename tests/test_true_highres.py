"""Test that use_true_highres controls whether match() uses actual highres pixels."""

import torch
import torch.nn.functional as F
from unittest.mock import patch, MagicMock
from romav2.romav2 import RoMaV2


def make_solid_image(value, h, w):
    """Create a (1, 3, H, W) image filled with `value`."""
    return torch.full((1, 3, h, w), value, dtype=torch.float32, device="cpu")


def test_highres_flag():
    lr_val = 0.2
    hr_val = 0.8
    img_lr_A = make_solid_image(lr_val, 100, 100)
    img_lr_B = make_solid_image(lr_val, 100, 100)
    img_hr_A = make_solid_image(hr_val, 200, 200)
    img_hr_B = make_solid_image(hr_val, 200, 200)

    for use_true_highres in [False, True]:
        captured = {}

        original_forward = RoMaV2.forward

        def capturing_forward(self, img_A_lr, img_B_lr, img_A_hr=None, img_B_hr=None, **kw):
            captured["img_A_hr"] = img_A_hr.clone() if img_A_hr is not None else None
            captured["img_B_hr"] = img_B_hr.clone() if img_B_hr is not None else None
            # Return dummy preds so match() doesn't crash
            B, C, H, W = img_A_hr.shape if img_A_hr is not None else img_A_lr.shape
            dummy = torch.zeros(B, H, W, 2)
            dummy_conf = torch.zeros(B, H, W, 4)
            return {
                "warp_AB": dummy, "confidence_AB": dummy_conf,
                "warp_BA": dummy, "confidence_BA": dummy_conf,
            }

        cfg = RoMaV2.Cfg(use_true_highres=use_true_highres, compile=False)

        # Build the object without __init__ (skip weight download)
        model = object.__new__(RoMaV2)
        torch.nn.Module.__init__(model)
        model.cfg = cfg
        model.H_lr = 80
        model.W_lr = 80
        model.H_hr = 160
        model.W_hr = 160
        model.bidirectional = True
        model.threshold = None
        model.training = False
        model.coarse_only = False

        with patch.object(type(model), "__call__", capturing_forward):
            model.match(
                img_lr_A, img_lr_B,
                im_A_high_res=img_hr_A,
                im_B_high_res=img_hr_B,
            )

        assert captured["img_A_hr"] is not None, "img_A_hr should not be None"
        mean_val = captured["img_A_hr"].mean().item()

        if use_true_highres:
            assert abs(mean_val - hr_val) < 0.05, (
                f"use_true_highres=True: expected HR pixels ~{hr_val}, got {mean_val:.3f}"
            )
        else:
            assert abs(mean_val - lr_val) < 0.05, (
                f"use_true_highres=False: expected upsampled LR pixels ~{lr_val}, got {mean_val:.3f}"
            )

        print(f"  use_true_highres={use_true_highres}: img_A_hr mean={mean_val:.4f} — OK")

    print("PASSED")


if __name__ == "__main__":
    test_highres_flag()
