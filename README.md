# INR Image Compression: Batch Size Effects

Bachelor's thesis project investigating how batch size affects training dynamics and reconstruction quality in Implicit Neural Representations (INR) for image compression.

## Project Structure

```
thesis-inr/
├── models/
│   ├── siren.py              # SIREN architecture (sinusoidal activations)
│   └── coin.py               # COIN architecture (ReLU + positional encoding)
├── training/
│   ├── sampling.py            # Pixel sampling strategies (full, random, patch)
│   └── trainer.py             # Training loop (standard + adaptive + time-budget)
├── evaluation/
│   ├── metrics.py             # PSNR, SSIM, image reconstruction
│   └── spectral.py            # FFT-based spectral analysis
├── experiments/
│   ├── run_sweep.py           # SIREN batch-size sweep (Experiment 1)
│   ├── run_extended.py        # COIN sweep, iso-time, adaptive, visuals, VRAM
│   ├── run_theory.py          # Gradient variance, LR scaling, difficulty
│   ├── run_seeds_and_jpeg.py  # Multi-seed validation + JPEG baseline
│   ├── analyze_results.py     # Analysis for main sweep
│   └── analyze_extended.py    # Analysis for extended experiments
├── configs/
│   └── batch_sweep.yaml       # Experiment configuration
├── data/
│   ├── images/                # Kodak dataset (24 PNG images)
│   └── download_kodak.py      # Dataset download script
├── submission/
│   ├── main.tex               # LaTeX thesis
│   ├── references.bib         # Bibliography
│   └── figures/               # All thesis figures
├── requirements.txt           # Python dependencies
├── RESEARCH_LOG.md            # Step-by-step research log
└── GUIDE.md                   # Code explanation and review
```

## Setup

### Requirements
- Python 3.10+
- NVIDIA GPU with CUDA support (tested on RTX 5060 Ti, CUDA 12.8)

### Installation

```bash
# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install -r requirements.txt
```

### Dataset

Download the Kodak PhotoCD dataset (24 images, ~12 MB):

```bash
python data/download_kodak.py
```

## Running Experiments

All experiments are run from the project root directory.

### Experiment 1: SIREN Batch-Size Sweep (120 runs, ~2.5 hours)
```bash
python -m experiments.run_sweep
```
Sweeps batch sizes [256, 1024, 4096, 16384, full] across all 24 Kodak images. Results saved to `experiments/results/`.

### Experiment 2-5: Extended Experiments
```bash
# All extended experiments (~8 hours total)
python -m experiments.run_extended --experiment all

# Or run individually:
python -m experiments.run_extended --experiment coin_sweep    # ~2.5 hours
python -m experiments.run_extended --experiment iso_time      # ~5 hours
python -m experiments.run_extended --experiment adaptive      # ~2 hours
python -m experiments.run_extended --experiment visuals       # ~30 min
python -m experiments.run_extended --experiment vram          # ~5 min
```

### Experiment 6-8: Theory Experiments
```bash
# All theory experiments (~2.5 hours)
python -m experiments.run_theory --experiment all

# Or individually:
python -m experiments.run_theory --experiment grad_variance   # ~1 hour
python -m experiments.run_theory --experiment lr_scaling       # ~1.5 hours
python -m experiments.run_theory --experiment difficulty       # instant
```

### Experiment 9-10: JPEG Comparison + Multi-Seed
```bash
python -m experiments.run_seeds_and_jpeg --experiment all     # ~1.5 hours
```

### Analysis
```bash
python -m experiments.analyze_results      # Aggregate main sweep
python -m experiments.analyze_extended     # Aggregate extended experiments
```

## Total Compute

- **Total training runs**: ~1,500
- **Total GPU time**: ~20 hours on NVIDIA RTX 5060 Ti
- **Hardware**: NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM, CUDA 12.8
- **Software**: Python 3.14, PyTorch 2.10

## Key Results

| Batch Size | PSNR (dB) | SSIM | Time (s) |
|:----------:|:---------:|:----:|:--------:|
| 256        | 21.72     | 0.525| 26.3     |
| 16384      | 28.56     | 0.810| 30.6     |
| Full       | 34.19     | 0.920| 231.9    |

Under time constraints (30s), bs=16384 outperforms full-batch by 2.7 dB. An adaptive schedule (50% bs=4096, 50% full) closes to within 1.8 dB of full-batch quality in 56% of the time.

## Thesis

The LaTeX source of the thesis is in `submission/`. Compile with:
```bash
cd submission
pdflatex main.tex && biber main && pdflatex main.tex && pdflatex main.tex
```
