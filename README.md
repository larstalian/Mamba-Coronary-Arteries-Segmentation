# Coronary Artery Segmentation with SegMamba

Resume-ready project for 3D coronary artery segmentation on ASOCA CTCA volumes using the exact SegMamba architecture from the original paper and official codebase.

## What This Repository Contains

- Reproducible training entrypoint with CLI config (`main.py`)
- Sliding-window inference script (`inference.py`)
- Lightweight volume visualization utility (`visualize.py`)
- SegMamba model implementation (`model_segmamba/segmamba.py`)
- Legacy experiment scripts kept for traceability (`legacy/`)
- Updated short report assets (`docs/`)

## Quick Start

### 1) Install dependencies (uv)

```bash
uv sync
```

Alternative:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2) Expected dataset layout (ASOCA)

```text
<DATA_ROOT>/
  Diseased/
    CTCA/Diseased_1.nrrd ... Diseased_19.nrrd
    Annotations/Diseased_1.nrrd ... Diseased_19.nrrd
  Normal/
    CTCA/Normal_1.nrrd ... Normal_19.nrrd
    Annotations/Normal_1.nrrd ... Normal_19.nrrd
```

### 3) Train

```bash
uv run python main.py \
  --data-root /path/to/asoca \
  --output-dir outputs/segmamba_run \
  --epochs 120 \
  --patch-size 224,224,96 \
  --samples-per-volume 3 \
  --val-interval 5 \
  --amp
```

### 4) Inference

```bash
uv run python inference.py \
  --checkpoint outputs/segmamba_run/checkpoints/best_model.pt \
  --image /path/to/Diseased_1.nrrd \
  --output outputs/segmamba_run/inference_mask.nrrd \
  --patch-size 224,224,96
```

### 5) Visualize slices

```bash
uv run python visualize.py --volume outputs/segmamba_run/inference_mask.nrrd
```

## Historical Results (from prior report)

These are validation-set numbers from the earlier project phase:

| Model | Mean Validation Dice |
|---|---:|
| UNet baseline | 0.7844 |
| Mamba-Encoder | 0.7734 |
| SegMamba | 0.7673 |

See the report in [`docs/report_2026.pdf`](docs/report_2026.pdf).

## Reports

- Updated short report PDF: [`docs/report_2026.pdf`](docs/report_2026.pdf)
- LaTeX source: [`docs/report_2026.tex`](docs/report_2026.tex)

## Notes

- `legacy/` contains older scripts retained for experiment traceability.
- Checkpoints and generated artifacts are ignored via `.gitignore`.
