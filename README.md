# Coronary Arteries Segmentation

SegMamba-based implementation for coronary artery segmentation.

## Dependencies

- PyTorch
- MONAI
- pynrrd
- tensorboard

## Setup

1. Clone [SegMamba](https://github.com/ge-xing/SegMamba) and install dependencies
2. Clone this repository
3. Update `root_dir` in `main.py` to your dataset path (expects ASOCA format)

## Usage

**Training:**
```bash
python main.py
```

**Inference:**
```bash
python inference.py
```

**Visualization:**
```bash
python visualize.py
```

## Training Configuration

- Epochs: 250
- Batch size: 1
- Learning rate: 0.001
- Patch size: 512×512×224
- Metrics: Dice coefficient, Hausdorff distance (95th percentile)

## Structure

- `model_segmamba/` - SegMamba model implementation
- `main.py` - Training script (ASOCA dataset, 512×512×224 patches)
- `inference.py` - Inference script
- `visualize.py` - Visualization utilities

## Additional Models

For UNet and Mamba-Encoder variants, see [LightM-UNet](https://github.com/MrBlankness/LightM-UNet).
