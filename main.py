#!/usr/bin/env python3
"""Train SegMamba on the ASOCA coronary artery dataset.

This script is designed as the primary, reproducible training entrypoint for the
repository. It supports deterministic split creation, targeted foreground patch
sampling, mixed-precision training, checkpointing, and JSONL metric logging.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from monai.losses import DiceLoss
from monai.metrics import HausdorffDistanceMetric
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    SpatialPadd,
)
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model_segmamba.segmamba import SegMamba


def parse_int_tuple(value: str, expected_len: int = 3) -> Tuple[int, ...]:
    parts = tuple(int(x.strip()) for x in value.split(","))
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(
            f"Expected {expected_len} comma-separated ints, got: {value}"
        )
    return parts


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_case_list(data_root: Path) -> List[Dict[str, str]]:
    cases: List[Dict[str, str]] = []
    groups = [
        ("Diseased", "Diseased"),
        ("Normal", "Normal"),
    ]

    for group_dir, prefix in groups:
        for idx in range(1, 20):
            image = data_root / group_dir / "CTCA" / f"{prefix}_{idx}.nrrd"
            label = data_root / group_dir / "Annotations" / f"{prefix}_{idx}.nrrd"
            if image.exists() and label.exists():
                cases.append({"image": str(image), "label": str(label)})

    if not cases:
        raise FileNotFoundError(
            f"No ASOCA cases found under {data_root}. Check dataset path/layout."
        )

    return cases


def split_cases(
    cases: Sequence[Dict[str, str]], val_ratio: float, seed: int
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be in (0, 1)")

    shuffled = list(cases)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_cases = shuffled[:val_count]
    train_cases = shuffled[val_count:]

    if not train_cases:
        raise ValueError("Validation split too large; no training cases left")

    return train_cases, val_cases


class ASOCAPatchDataset(Dataset):
    def __init__(
        self,
        cases: Sequence[Dict[str, str]],
        patch_size: Tuple[int, int, int],
        samples_per_volume: int,
        foreground_sampling_prob: float,
    ) -> None:
        self.cases = list(cases)
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.foreground_sampling_prob = foreground_sampling_prob
        self.transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityd(keys=["image"]),
                SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
                EnsureTyped(keys=["image", "label"]),
            ]
        )

    def __len__(self) -> int:
        return len(self.cases) * self.samples_per_volume

    def _choose_patch_start(
        self, label: torch.Tensor
    ) -> Tuple[int, int, int]:
        spatial_shape = label.shape[-3:]
        patch = self.patch_size

        starts: List[int] = [0, 0, 0]

        use_foreground = random.random() < self.foreground_sampling_prob
        fg = torch.nonzero(label[0] > 0, as_tuple=False)

        center: Sequence[int]
        if use_foreground and len(fg) > 0:
            center = fg[random.randrange(len(fg))].tolist()
        else:
            center = [random.randrange(d) for d in spatial_shape]

        for axis in range(3):
            dim = int(spatial_shape[axis])
            size = int(patch[axis])
            if dim <= size:
                starts[axis] = 0
                continue

            min_start = 0
            max_start = dim - size
            proposed = int(center[axis] - size // 2)
            starts[axis] = max(min_start, min(proposed, max_start))

        return tuple(starts)

    @staticmethod
    def _crop(
        tensor: torch.Tensor, start: Tuple[int, int, int], size: Tuple[int, int, int]
    ) -> torch.Tensor:
        slices = [slice(None)]
        for axis in range(3):
            s = start[axis]
            e = s + size[axis]
            slices.append(slice(s, e))
        return tensor[tuple(slices)]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case = self.cases[idx % len(self.cases)]
        sample = self.transforms(case)
        image = sample["image"].float()
        label = sample["label"].float()

        start = self._choose_patch_start(label)
        image_patch = self._crop(image, start, self.patch_size)
        label_patch = self._crop(label, start, self.patch_size)

        # Ensure binary target map for loss/metrics.
        label_patch = (label_patch > 0).float()

        return {"image": image_patch, "label": label_patch}


def dice_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = (torch.sigmoid(logits) > 0.5).float()
    inter = (preds * targets).sum(dim=(1, 2, 3, 4))
    union = preds.sum(dim=(1, 2, 3, 4)) + targets.sum(dim=(1, 2, 3, 4))
    return ((2.0 * inter + 1e-5) / (union + 1e-5)).mean()


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    dice_loss_fn: DiceLoss,
    dice_weight: float,
    bce_weight: float,
) -> torch.Tensor:
    dice_loss = dice_loss_fn(logits, targets)
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
    return dice_weight * dice_loss + bce_weight * bce_loss


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dice_loss_fn: DiceLoss,
    amp_enabled: bool,
    dice_weight: float,
    bce_weight: float,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    steps = 0

    hd95_metric = HausdorffDistanceMetric(percentile=95, include_background=False)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with autocast(enabled=amp_enabled):
                logits = model(images)
                loss = combined_loss(
                    logits,
                    labels,
                    dice_loss_fn=dice_loss_fn,
                    dice_weight=dice_weight,
                    bce_weight=bce_weight,
                )

            preds = (torch.sigmoid(logits) > 0.5).float()
            hd95_metric(y_pred=preds, y=labels)

            total_loss += float(loss.item())
            total_dice += float(dice_from_logits(logits, labels).item())
            steps += 1

    mean_loss = total_loss / max(steps, 1)
    mean_dice = total_dice / max(steps, 1)

    hd95_value = hd95_metric.aggregate().item()
    hd95_metric.reset()

    return {
        "loss": mean_loss,
        "dice": mean_dice,
        "hd95": float(hd95_value),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_val_dice: float,
    patch_size: Tuple[int, int, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_dice": best_val_dice,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "model_kwargs": {
                "in_chans": 1,
                "out_chans": 1,
                "depths": [2, 2, 2, 2],
                "feat_size": [48, 96, 192, 384],
                "hidden_size": 768,
            },
            "patch_size": list(patch_size),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SegMamba on ASOCA")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/segmamba"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--samples-per-volume", type=int, default=3)
    parser.add_argument("--foreground-sampling-prob", type=float, default=0.8)
    parser.add_argument("--patch-size", type=parse_int_tuple, default=(224, 224, 96))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dice-weight", type=float, default=0.8)
    parser.add_argument("--bce-weight", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    amp_enabled = args.amp and device.type == "cuda"

    output_dir = args.output_dir
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    all_cases = build_case_list(args.data_root)
    train_cases, val_cases = split_cases(all_cases, args.val_ratio, args.seed)

    train_ds = ASOCAPatchDataset(
        train_cases,
        patch_size=args.patch_size,
        samples_per_volume=args.samples_per_volume,
        foreground_sampling_prob=args.foreground_sampling_prob,
    )
    val_ds = ASOCAPatchDataset(
        val_cases,
        patch_size=args.patch_size,
        samples_per_volume=max(1, args.samples_per_volume // 2),
        foreground_sampling_prob=1.0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SegMamba(
        in_chans=1,
        out_chans=1,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        hidden_size=768,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.92)
    scaler = GradScaler(enabled=amp_enabled)
    dice_loss_fn = DiceLoss(sigmoid=True)

    print(f"Using device: {device}")
    print(f"Cases: total={len(all_cases)} train={len(train_cases)} val={len(val_cases)}")
    print(f"Patch size: {args.patch_size} | Samples/volume: {args.samples_per_volume}")

    metrics_file = output_dir / "metrics.jsonl"
    best_val_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        steps = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp_enabled):
                logits = model(images)
                loss = combined_loss(
                    logits,
                    labels,
                    dice_loss_fn=dice_loss_fn,
                    dice_weight=args.dice_weight,
                    bce_weight=args.bce_weight,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_dice = dice_from_logits(logits.detach(), labels)
            epoch_loss += float(loss.item())
            epoch_dice += float(batch_dice.item())
            steps += 1

            progress.set_postfix(loss=f"{loss.item():.4f}", dice=f"{batch_dice.item():.4f}")

        scheduler.step()

        train_loss = epoch_loss / max(steps, 1)
        train_dice = epoch_dice / max(steps, 1)

        record: Dict[str, float | int | None] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": None,
            "val_dice": None,
            "val_hd95": None,
            "lr": optimizer.param_groups[0]["lr"],
        }

        should_validate = epoch == 1 or (epoch % args.val_interval == 0)
        if should_validate:
            val_metrics = evaluate(
                model,
                val_loader,
                device=device,
                dice_loss_fn=dice_loss_fn,
                amp_enabled=amp_enabled,
                dice_weight=args.dice_weight,
                bce_weight=args.bce_weight,
            )
            record["val_loss"] = val_metrics["loss"]
            record["val_dice"] = val_metrics["dice"]
            record["val_hd95"] = val_metrics["hd95"]

            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                save_checkpoint(
                    checkpoints_dir / "best_model.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_dice=best_val_dice,
                    patch_size=args.patch_size,
                )

        save_checkpoint(
            checkpoints_dir / "latest_model.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_dice=best_val_dice,
            patch_size=args.patch_size,
        )

        with metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(json.dumps(record))

    print(f"Training finished. Checkpoints saved in: {checkpoints_dir}")


if __name__ == "__main__":
    main()
