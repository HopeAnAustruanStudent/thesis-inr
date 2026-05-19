# Research Log: INR Image Compression — Batch Size Study

## Project Overview

**Topic**: Implicit Neural Representations (INR) for Image Compression
**Research question**: How does batch size affect training dynamics, reconstruction quality, and spectral fidelity in INR-based image compression — and does this depend on the network architecture?
**Dataset**: Kodak PhotoCD (24 images, 768×512)
**Architectures**: SIREN (sinusoidal activations) and COIN (ReLU + positional encoding)

---

## Step-by-Step: What We Did and Why

### Phase 1: Core Implementation

#### Step 1. SIREN Model (`models/siren.py`)
**What**: Implemented the SIREN architecture — an MLP where each hidden layer uses `sin(ω₀ · Wx + b)` instead of ReLU.
**Why**: SIREN is the canonical INR architecture (Sitzmann et al., NeurIPS 2020). Its sinusoidal activations overcome the "spectral bias" problem — ReLU networks tend to learn low-frequency functions first and struggle with high-frequency details.
**Key details**:
- 5 hidden layers, 256 units each, ω₀=30
- Custom weight initialization from the paper (critical for stability)
- Sigmoid output to map to [0,1] RGB range
- 264,707 total parameters

#### Step 2. Pixel Samplers (`training/sampling.py`)
**What**: Three strategies for feeding pixel data to the model during training:
1. `full_image` — use ALL pixels every step (deterministic gradient)
2. `random_pixels(n)` — sample n random pixels (stochastic gradient)
3. `grid_patch(size)` — sample a random square patch

**Why**: The batch size question IS the sampling question. In standard deep learning, you sample from a dataset of independent examples. Here, we sample from pixels within a SINGLE image — a fundamentally different regime. The pixels are spatially correlated, and the sampling strategy determines what frequency information the gradient "sees" at each step.

#### Step 3. Trainer (`training/trainer.py`)
**What**: Training loop that overfits a model to one image.
**Why**: INR compression works by memorizing a single image as network weights. The training loop is the "encoding" step. We use Adam optimizer (lr=1e-4) with MSE loss. The trainer now supports:
- Fixed-step training (standard)
- Time-budget training (for iso-time experiments)
- Wall-clock PSNR logging (for convergence analysis)

#### Step 4. Evaluation Metrics (`evaluation/metrics.py`)
**What**: PSNR (peak signal-to-noise ratio) and SSIM (structural similarity) computation, plus full-image reconstruction from a trained model.
**Why**: PSNR measures pixel-level fidelity in dB — the standard metric for compression quality. SSIM captures perceptual similarity. Together they give both objective and perceptual quality measures.

#### Step 5. Spectral Analysis (`evaluation/spectral.py`)
**What**: FFT-based frequency analysis — compute 2D FFT of original and reconstructed images, split into radial frequency bands (low/mid/high), measure error per band.
**Why**: This is the KEY analysis that makes this thesis non-trivial. The question isn't just "does quality go down with smaller batches?" (obvious) — it's "WHICH frequencies are lost?" This reveals the mechanism: mini-batching introduces spatial undersampling, which disproportionately affects high-frequency components.

---

### Phase 2: SIREN Batch-Size Sweep (Experiment 1)

#### Step 6. Configuration & Runner (`configs/batch_sweep.yaml`, `experiments/run_sweep.py`)
**What**: Sweep over batch sizes [256, 1024, 4096, 16384, "full"] across all 24 Kodak images. 120 total training runs.
**Why**: This is the core experiment — measure how batch size affects quality, training time, and spectral fidelity.

#### Step 7. Results Analysis (`experiments/analyze_results.py`)
**What**: Aggregate results across images, compute mean±std, generate publication plots.
**Results** (averaged over 24 Kodak images):

| Batch Size | PSNR (dB) | SSIM | Time (s) | High-Freq Error |
|------------|-----------|------|----------|-----------------|
| 256 | 21.72±2.34 | 0.525 | 26.3 | 4.420 |
| 1024 | 23.69±2.52 | 0.622 | 26.3 | 3.974 |
| 4096 | 25.75±2.72 | 0.713 | 25.7 | 2.411 |
| 16384 | 28.56±2.96 | 0.810 | 30.6 | 0.891 |
| Full | 34.19±2.82 | 0.920 | 231.9 | 0.087 |

**Key finding**: PSNR increases monotonically with batch size. High-frequency spectral error drops 50x from bs=256 to full-batch. Mini-batch sizes 256–4096 have nearly identical wall-clock time (~26s) — the overhead is in the per-step forward/backward pass, not data loading.

---

### Phase 3: Extended Experiments (PhD-Level Rigor)

**Problem with Phase 2**: The comparison is UNFAIR. At 2000 steps:
- bs=256 sees 256 × 2000 = 512K pixels total
- full-batch sees 393K × 2000 = 786M pixels total — **1500x more data**

Of course full-batch wins. The question is: is it the batch size per se, or just seeing more data?

#### Step 8. COIN Model (`models/coin.py`)
**What**: COIN architecture — ReLU MLP with Fourier positional encoding.
**Why**: SIREN and COIN encode frequency information differently:
- SIREN: frequencies are in the ACTIVATIONS (sin)
- COIN: frequencies are in the INPUT (positional encoding)

If batch size affects them differently, this reveals something fundamental about how architectures interact with the optimization landscape.

#### Step 9. Iso-Time Experiment
**What**: Train both SIREN and COIN with fixed time budgets (30s, 60s, 120s) at each batch size.
**Why**: This is the FAIR comparison. Same GPU time → which batch size gives best quality? A practitioner cares about "I have 60 seconds, what should I choose?" not "I have 2000 steps."

This also controls for the data-volume confound: smaller batches get MORE steps in the same time (faster per step), so they may see comparable total pixels.

#### Step 10. Adaptive Schedule
**What**: Start training with small batches (fast, learns coarse structure), then switch to large batches (slow, refines high-frequency details).
**Why**: This is our PRACTICAL CONTRIBUTION — not just measurement, but a method. If adaptive scheduling achieves near-full-batch quality at near-mini-batch speed, that's a publishable result.

Schedules tested:
- A: 30% at bs=1024 → 70% at full
- B: 50% at bs=4096 → 50% at full
- C: 20% at bs=256 → 30% at bs=4096 → 50% at full

#### Step 11. VRAM Profiling
**What**: Measure peak GPU memory for each batch size and architecture.
**Why**: Memory is a REAL constraint. Full-batch on a 4K image may not fit in consumer GPU VRAM.

Results from kodim01 (768×512):

| Batch Size | SIREN VRAM | COIN VRAM |
|------------|------------|-----------|
| 256 | 23 MB | 23 MB |
| 1024 | 33 MB | 29 MB |
| 4096 | 69 MB | 53 MB |
| 16384 | 213 MB | 148 MB |
| Full | 4646 MB | 2786 MB |

Full-batch uses 4.6 GB for SIREN — this would NOT fit on a 4GB GPU. For 4K images, it would exceed even 16GB cards. This makes batch size a practical necessity, not just an optimization choice.

#### Step 12. Visual Reconstruction Grid
**What**: Side-by-side images showing original + reconstructions at each batch size for both architectures.
**Why**: Numbers (PSNR/SSIM) don't tell the full story. Visual inspection reveals WHERE quality degrades — edges, textures, flat regions.

---

### Phase 4: Report & Presentation

#### Iso-Time Results (Key Finding!)

| Budget | SIREN bs=16384 | SIREN full | COIN bs=16384 | COIN full |
|--------|---------------|------------|---------------|-----------|
| 30s  | **28.69 dB** | 25.73 dB | 23.91 dB | 19.72 dB |
| 60s  | **29.93 dB** | 28.21 dB | 25.70 dB | 21.58 dB |
| 120s | 30.87 dB | **30.97 dB** | 28.13 dB | 23.94 dB |

**This is the strongest thesis finding**: under time constraints, full-batch is NOT optimal. bs=16384 dominates up to ~2 minutes because it gets ~8× more gradient steps per second. Full-batch only wins when given enough time to compensate with its exact gradients.

**Architecture interaction**: COIN's full-batch is consistently WORST — it needs many steps to converge through its ReLU landscape. SIREN handles few-step full-batch better thanks to sinusoidal activations providing richer gradient signals per step.

#### Step 13. LaTeX Report (`report/main.tex`)
Standard thesis structure with actual data filled in: abstract, introduction, related work, methodology, experiments, results, discussion, conclusion. Includes all figures and tables from experiments.

#### Step 14. Marp Slides (`slides/slides.md`)
12-slide presentation with results, plots, and key findings.

---

## Architecture of the Codebase

```
thesis-inr/
├── models/
│   ├── siren.py          SIREN: sin activations, ω₀=30
│   └── coin.py           COIN: ReLU + positional encoding
├── training/
│   ├── sampling.py       3 sampling strategies (full, random, patch)
│   └── trainer.py        Trainer + AdaptiveTrainer
├── evaluation/
│   ├── metrics.py        PSNR, SSIM, reconstruction
│   └── spectral.py       FFT analysis, band errors, plots
├── experiments/
│   ├── run_sweep.py      Original SIREN sweep
│   ├── run_extended.py   COIN + iso-time + adaptive + visuals + VRAM
│   ├── analyze_results.py    Analysis for sweep
│   └── analyze_extended.py   Analysis for extended experiments
├── configs/
│   └── batch_sweep.yaml  Experiment configuration
├── data/
│   ├── images/           Kodak dataset (24 PNG)
│   └── download_kodak.py Download script
├── report/
│   └── main.tex          LaTeX thesis
├── slides/
│   └── slides.md         Marp presentation
└── requirements.txt      Dependencies
```

## How Each File Works

### `models/siren.py`
- `SineLayer`: one linear layer → multiply by ω₀ → apply sin()
- `SIREN`: stack of SineLayer + final Linear+Sigmoid
- Weight init: first layer U[-1/n, 1/n], rest U[-√(6/n)/ω₀, √(6/n)/ω₀]
- Forward: coords (N,2) → RGB (N,3)

### `models/coin.py`
- `PositionalEncoding`: maps (x,y) → [sin(2⁰πx), cos(2⁰πx), ..., sin(2⁹πx), cos(2⁹πx)]
- This expands 2 inputs to 40 features (2 dims × 10 freqs × 2 for sin/cos)
- `COIN`: PositionalEncoding → stack of Linear+ReLU → Linear+Sigmoid
- Kaiming initialization for ReLU layers

### `training/sampling.py`
- `_build_coord_grid(H, W)`: creates (H×W, 2) array of (x,y) in [-1,1]
- `full_image()`: returns ALL coordinates and colors
- `random_pixels(n)`: random subset of n pixels (numpy random choice)
- `grid_patch(size)`: random square crop → coordinates in global space
- `get_sampler()`: factory function that returns a callable

### `training/trainer.py`
- `Trainer`: standard training loop
  - Supports step-based OR time-based stopping
  - Logs loss, PSNR per step, PSNR vs wall-clock time
- `AdaptiveTrainer`: changes batch size mid-training
  - Schedule: list of (fraction, batch_size) pairs
  - Progress-based phase switching

### `evaluation/metrics.py`
- `psnr()`: 10 × log10(1/MSE) between two images
- `ssim()`: wraps skimage structural_similarity
- `reconstruct()`: queries model at every pixel, chunks to avoid OOM
- `evaluate()`: reconstruct + compute PSNR + SSIM

### `evaluation/spectral.py`
- `fft_magnitude()`: convert to grayscale → 2D FFT → center → log magnitude
- `radial_profile()`: average magnitude at each radial frequency
- `spectral_error_by_band()`: split radial axis into low/mid/high → MSE per band
- `plot_spectral_comparison()`: two-panel plot (profiles + bar chart)

---

## Key Concepts for Understanding

### What is an INR?
An image is normally stored as a grid of pixels (H×W×3). An INR stores it as a function f(x,y) → (r,g,b). The function is a neural network. To "decode" the image, you evaluate f at every pixel coordinate.

### Why batch size matters for INR
In normal deep learning, mini-batching samples from INDEPENDENT examples. In INR, we sample pixels from ONE image — they're spatially correlated. A small random batch may miss entire regions, giving the gradient incomplete spatial information. This is especially harmful for high-frequency details (edges, textures) which require dense spatial coverage to represent.

### Spectral bias
Neural networks learn low frequencies first, then high frequencies. This is called "spectral bias." SIREN was designed to overcome this with sinusoidal activations. But batch size introduces ANOTHER source of spectral bias: spatial undersampling in mini-batches effectively low-pass filters the gradient signal.

### Compression ratio
Model parameters = 264,707. In float32: 264,707 × 4 = 1,058,828 bytes ≈ 1.01 MB.
Kodak image (768×512×3): 1,179,648 bytes ≈ 1.15 MB.
Ratio: 1.15/1.01 ≈ 1.13× compression.
With float16: ~2.26×. With quantization: even higher.

---

## Phase 5: Additional Experiments

#### JPEG Baseline Comparison
SIREN at float16 (517 KB) achieves 34.19 dB. JPEG at q=50 (120 KB) achieves 34.37 dB. JPEG wins on raw rate-distortion. INR's advantages are elsewhere: continuous resolution, differentiability, functional representation.

#### Multi-Seed Validation (3 seeds × 6 images)
Inter-seed variance is <0.02 for all mini-batch sizes (vs inter-image variance ~8). Results are rock-solid — random initialization doesn't matter. Full-batch has slightly higher seed variance (0.79) — deterministic gradients let the optimizer reach different local minima per init.

#### Gradient Variance Analysis
| bs | Gradient CV | PSNR |
|:--:|:-----------:|:----:|
| 256 | 0.170 (17%) | 21.72 |
| full | 0.000 (exact) | 34.19 |

This is the causal mechanism: noisy gradients prevent stable high-frequency learning.

#### LR Scaling — Standard Fix Fails
Best rule (sqrt) recovers only 0.9 dB of the 12 dB gap. The problem is spatial information loss, not learning rate.

#### Per-Image Difficulty
r = -0.07 correlation with edge density. Essentially a null result — batch size effect is universal across all image types.

---

## Phase 6: Professor Review & Improvements

### Review Summary (MIT-level assessment)

**Grade: A- for bachelor's thesis**

**Strongest contributions identified:**
1. Iso-time reversal (bs=16384 beats full-batch under time constraints)
2. Gradient variance mechanism (17% CV at bs=256 vs 0% at full)
3. LR scaling failure (standard DL recipes don't apply to INR)

**Critical issues to fix:**
1. **"95% quality" claim is misleading** — PSNR is log scale. 32.39 vs 34.18 dB means adaptive B has 53% MORE error (MSE) than full-batch. Must report distortion ratio.
2. **Table 1 is unfair without caveat** — comparing bs=256 at 2000 steps vs full at 2000 steps conflates batch size with data volume. Need immediate caveat pointing to iso-time.
3. **COIN comparison fairness** — COIN designed for meta-learning, not single-image overfitting. Must state explicitly.
4. **Contributions list doesn't match findings** — Introduction lists generic contributions but actual strongest findings are iso-time reversal and gradient variance.
5. **No formal problem statement** — need min_θ Σ ||f_θ(x_i) - y_i||² equation.
6. **Related work too thin** — missing WIRE, weight quantization, sampling strategies.
7. **Conclusion says "over 600 runs"** — actually ~1,500.
8. **Difficulty analysis is a null result** — frame it honestly.
9. **~3,400 words is short** — most institutions expect 8,000-15,000.

**Defense questions to prepare for:**
1. Fixed-step vs iso-time: which is the "real" result? (Both — different questions)
2. Gradient variance: how do you know it specifically affects high frequencies? (Indirect evidence — acknowledged as limitation)
3. Adaptive schedule: would 50/50 work at 4K? (Unknown — flag as limitation)
4. Why use INR over JPEG? (Continuous representation, differentiability)
5. Does batch size effect persist for different model sizes? (Unknown — flag as limitation)
