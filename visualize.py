#!/usr/bin/env python3
"""Visualize orthogonal slices from a 3D NRRD volume."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nrrd
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize NRRD slices")
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cmap", type=str, default="gray")
    return parser.parse_args()


def normalize_slice(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    xmin, xmax = float(x.min()), float(x.max())
    if xmax <= xmin:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def main() -> None:
    args = parse_args()
    volume, _ = nrrd.read(str(args.volume))

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")

    z, y, x = volume.shape
    z_mid, y_mid, x_mid = z // 2, y // 2, x // 2

    axial = normalize_slice(volume[z_mid, :, :])
    coronal = normalize_slice(volume[:, y_mid, :])
    sagittal = normalize_slice(volume[:, :, x_mid])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(axial, cmap=args.cmap)
    axes[0].set_title(f"Axial z={z_mid}")
    axes[1].imshow(coronal, cmap=args.cmap)
    axes[1].set_title(f"Coronal y={y_mid}")
    axes[2].imshow(sagittal, cmap=args.cmap)
    axes[2].set_title(f"Sagittal x={x_mid}")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output, dpi=180, bbox_inches="tight")
        print(f"Saved visualization to: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
