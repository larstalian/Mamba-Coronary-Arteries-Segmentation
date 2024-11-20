#!/usr/bin/env python3
"""Run SegMamba inference on a 3D CT volume."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import nrrd
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
)

from model_segmamba.segmamba import SegMamba


def parse_int_tuple(value: str, expected_len: int = 3) -> Tuple[int, ...]:
    parts = tuple(int(x.strip()) for x in value.split(","))
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(
            f"Expected {expected_len} comma-separated ints, got: {value}"
        )
    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with SegMamba")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prob-output", type=Path, default=None)
    parser.add_argument("--patch-size", type=parse_int_tuple, default=(224, 224, 96))
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> SegMamba:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        model_kwargs = checkpoint.get(
            "model_kwargs",
            {
                "in_chans": 1,
                "out_chans": 1,
                "depths": [2, 2, 2, 2],
                "feat_size": [48, 96, 192, 384],
                "hidden_size": 768,
            },
        )
    else:
        state_dict = checkpoint
        model_kwargs = {
            "in_chans": 1,
            "out_chans": 1,
            "depths": [2, 2, 2, 2],
            "feat_size": [48, 96, 192, 384],
            "hidden_size": 768,
        }

    model = SegMamba(**model_kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityd(keys=["image"]),
            EnsureTyped(keys=["image"]),
        ]
    )

    sample = transforms({"image": str(args.image)})
    image = sample["image"].unsqueeze(0).to(device)

    model = load_model(args.checkpoint, device)

    with torch.no_grad():
        logits = sliding_window_inference(
            image,
            roi_size=args.patch_size,
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=args.overlap,
        )

    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (probs >= args.threshold).astype(np.uint8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nrrd.write(str(args.output), mask)

    if args.prob_output is not None:
        args.prob_output.parent.mkdir(parents=True, exist_ok=True)
        nrrd.write(str(args.prob_output), probs.astype(np.float32))

    print(f"Inference complete. Saved binary mask to: {args.output}")
    if args.prob_output is not None:
        print(f"Saved probability map to: {args.prob_output}")


if __name__ == "__main__":
    main()
