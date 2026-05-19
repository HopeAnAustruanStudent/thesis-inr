# Complete Guide: What We Did, Why, and How the Code Works

This document explains everything in simple terms so you can confidently present and defend this thesis.

---

## Part 1: The Big Picture

### What is this thesis about?

Normal images are stored as grids of pixels (like a spreadsheet of colors). We take a completely different approach: we train a tiny neural network to **memorize** an image. Instead of storing pixels, we store the network's weights. To see the image, we ask the network "what color is at position (x,y)?" for every pixel.

This is called an **Implicit Neural Representation (INR)**.

### What's the research question?

When training this network, we can show it all pixels at once ("full batch") or small random subsets ("mini-batch"). **How does this choice affect the quality of the memorized image?**

This sounds simple, but it turns out to be surprisingly deep — the answer depends on the architecture, the time budget, and the frequency content of the image.

### What did we find?

1. Bigger batches = better quality (not surprising)
2. But under time constraints, full-batch is actually WORSE than bs=16384 (surprising!)
3. The effect depends on which architecture you use (SIREN vs COIN)
4. We can explain WHY: gradient noise prevents learning fine details
5. Standard fixes (adjusting learning rate) don't help
6. A simple "start small, finish big" schedule gets 95% of the quality in 56% of the time

---

## Part 2: Step-by-Step What We Did

### Step 1: Built the SIREN model
**File**: `models/siren.py`
**What**: A neural network where each layer computes `sin(omega * (W*x + b))` instead of the usual `max(0, W*x + b)` (ReLU).
**Why**: Regular networks are bad at learning detailed patterns (high frequencies). The sin() activation solves this — it's called overcoming "spectral bias."
**Key number**: 264,707 parameters, ~1 MB stored as weights.

### Step 2: Built pixel samplers
**File**: `training/sampling.py`
**What**: Three ways to feed pixels to the network during training:
- `full_image`: ALL 393,216 pixels every step
- `random_pixels(n)`: n random pixels every step
- `grid_patch(size)`: a random square crop

**Why**: This is the core of our experiment — how we sample pixels IS the batch size question.

### Step 3: Built the training loop
**File**: `training/trainer.py`
**What**: Standard PyTorch training: forward pass, compute loss (MSE), backpropagate, update weights with Adam optimizer.
**Why**: The network "memorizes" the image by minimizing the difference between its predictions and actual pixel colors.

### Step 4: Built evaluation metrics
**File**: `evaluation/metrics.py`
**What**: PSNR (measures pixel accuracy in dB — higher is better) and SSIM (measures visual similarity — closer to 1 is better).
**Why**: We need numbers to compare different configurations.

### Step 5: Built spectral analysis
**File**: `evaluation/spectral.py`
**What**: Converts images to frequency domain using FFT (Fourier transform), then measures how much error there is at low, mid, and high frequencies.
**Why**: This reveals the MECHANISM — we don't just show "quality is worse" but show "specifically high-frequency details are lost."

### Step 6: Ran SIREN batch-size sweep
**File**: `experiments/run_sweep.py`
**Runs**: 24 images x 5 batch sizes = 120 training runs
**Result**: PSNR goes from 21.7 dB (bs=256) to 34.2 dB (full). High-freq error drops 50x.

### Step 7: Built COIN model
**File**: `models/coin.py`
**What**: Alternative architecture — uses ReLU activations but adds "positional encoding" to the input (converts (x,y) into 40 sine/cosine features).
**Why**: To test if the batch size effect is universal or architecture-specific.

### Step 8: Ran COIN sweep
**File**: `experiments/run_extended.py`
**Result**: COIN is worse at every batch size. The gap WIDENS from 1.1 dB to 9.1 dB. This means SIREN benefits MORE from large batches.

### Step 9: Ran iso-time experiment
**What**: Instead of giving every config the same number of steps, give them the same WALL CLOCK TIME (30s, 60s, 120s).
**Why**: The original experiment is unfair — full-batch sees 1500x more pixels. Same time = fair comparison.
**Key finding**: At 30s, bs=16384 gets 28.5 dB but full-batch only gets 25.8 dB. Full-batch is too slow per step!

### Step 10: Ran adaptive schedule experiment
**What**: Start with small batches (fast, learn rough shape) then switch to full batch (slow, learn fine details).
**Result**: 50% at bs=4096 then 50% at full = 32.4 dB in 131s (vs 34.2 dB in 234s for pure full-batch).

### Step 11: Measured gradient variance
**What**: At each batch size, sample 30 different mini-batches and measure how much the gradient changes between them.
**Result**: At bs=256, gradient fluctuates 17%. At full-batch, 0% (deterministic). This NOISE is why small batches can't learn fine details.

### Step 12: Tested learning rate scaling
**What**: In standard deep learning, you can compensate for small batches by adjusting the learning rate. Does this work for INR?
**Result**: NO. Best rule (sqrt scaling) recovers only 0.9 dB of a 12 dB gap. The problem isn't the learning rate — it's that small batches literally don't contain enough spatial information.

### Step 13: Analyzed per-image difficulty
**What**: Do some images suffer more from small batches?
**Result**: No significant correlation (r=-0.07). ALL images are affected equally.

### Step 14: JPEG comparison
**What**: How does INR compare to JPEG?
**Result**: JPEG wins on pure file-size vs quality. SIREN at 517 KB = 34.2 dB. JPEG at 120 KB = 34.4 dB. But INR has unique advantages (continuous resolution, differentiability).

### Step 15: Multi-seed validation
**What**: Run everything 3 times with different random seeds.
**Result**: Inter-seed variance < 0.02, inter-image variance ~ 8. Our results are rock solid — not dependent on lucky initialization.

---

## Part 3: Code Review — How Each File Works

### `models/siren.py` (89 lines)

```python
class SineLayer(nn.Module):
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))
```

**How it works**:
- `SineLayer`: One linear transformation (matrix multiply + bias) followed by sin()
- `omega_0 = 30`: Controls the "frequency" of the sine — higher = can represent finer details
- `_init_weights()`: Critical! Uses special initialization from the SIREN paper. Wrong init = network doesn't train. First layer uses uniform [-1/n, 1/n], hidden layers use [-sqrt(6/n)/omega, sqrt(6/n)/omega]
- `SIREN`: Stacks 5 SineLayers + one final Linear layer with Sigmoid (to map output to [0,1] RGB range)
- `n_parameters()`: Returns total parameter count (264,707)

### `models/coin.py` (111 lines)

```python
class PositionalEncoding(nn.Module):
    def forward(self, x):
        scaled = x.unsqueeze(-1) * self.freqs  # (N, 2, 10)
        return torch.cat([sin(scaled), cos(scaled)], dim=-1)  # (N, 40)
```

**How it works**:
- `PositionalEncoding`: Takes (x,y) coordinates and expands them using sine/cosine at 10 different frequencies (2^0 to 2^9). This turns 2 numbers into 40 numbers.
- Why? ReLU networks can't learn high frequencies from raw coordinates. Positional encoding "pre-computes" frequency information.
- `COIN`: PositionalEncoding -> 5 layers of Linear+ReLU -> Linear+Sigmoid
- Uses Kaiming initialization (standard for ReLU networks)

### `training/sampling.py` (153 lines)

```python
def _build_coord_grid(H, W):
    xs = np.linspace(-1, 1, W)  # column coordinates
    ys = np.linspace(-1, 1, H)  # row coordinates
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
```

**How it works**:
- `_build_coord_grid`: Creates a grid of (x,y) coordinates covering [-1,1] x [-1,1]. For a 768x512 image, this is 393,216 coordinate pairs.
- `full_image`: Returns ALL coordinates and their colors. The batch = entire image.
- `random_pixels(n)`: Uses `np.random.choice` to pick n random pixel indices, returns those coordinates and colors.
- `grid_patch(size)`: Picks a random top-left corner, extracts a size x size square patch.
- `get_sampler`: Factory function — takes a mode string ("full_image", "random_pixels", "grid_patch") and returns the appropriate function.

### `training/trainer.py` (230 lines)

```python
def train(self):
    for step in range(n_steps):
        coords, colors = self.sampler(self.image)  # get batch
        pred = self.model(coords)                   # forward pass
        loss = self.criterion(pred, colors)          # MSE loss
        loss.backward()                              # compute gradients
        self.optimizer.step()                        # update weights
```

**How it works**:
- `Trainer`: Standard training loop. Supports both step-based and time-based stopping.
- `psnr_vs_time`: Logs PSNR with wall-clock timestamps (critical for iso-time analysis).
- `AdaptiveTrainer`: Same loop but switches sampler mid-training based on a schedule. Uses `_get_phase(progress)` to determine which batch size to use at each point.

### `evaluation/metrics.py` (106 lines)

```python
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 10 * log10(1 / mse)
```

**How it works**:
- `psnr`: Computes mean squared error, converts to dB. 30 dB = good, 40 dB = excellent, 20 dB = blurry.
- `ssim`: Wraps scikit-image's structural_similarity function.
- `reconstruct`: Feeds ALL coordinates through the model in chunks of 65536 (to avoid out-of-memory), reassembles into H x W x 3 image.
- `evaluate`: Combines reconstruct + psnr + ssim into one call.

### `evaluation/spectral.py` (152 lines)

```python
def fft_magnitude(img):
    gray = to_grayscale(img)
    f = np.fft.fft2(gray)           # 2D Fourier transform
    fshift = np.fft.fftshift(f)     # center low frequencies
    return np.log1p(np.abs(fshift)) # log magnitude
```

**How it works**:
- `fft_magnitude`: Converts image to grayscale, computes 2D FFT, centers it (low freq in middle, high freq at edges), takes log of magnitude.
- `radial_profile`: Averages the FFT magnitude in concentric rings from center (low freq) to edge (high freq). Produces a 1D curve.
- `spectral_error_by_band`: Splits the radial frequency axis into 3 equal bands (low/mid/high), computes MSE between original and reconstructed profiles in each band.

### `experiments/run_sweep.py` (251 lines)

**How it works**:
- Loads config from YAML, loads images, loops over batch sizes
- For each: creates model, trains, evaluates, saves weights + spectral plots
- Saves lightweight results to JSON (strips loss history to keep file small)
- Generates per-image summary plots and loss curve overlays

### `experiments/run_extended.py` (310 lines)

**5 experiments in one file**:
1. `run_coin_sweep`: Same as SIREN sweep but creates COIN models
2. `run_iso_time`: Uses `time_budget_s` parameter in Trainer
3. `run_adaptive`: Uses AdaptiveTrainer with 3 different schedules + 3 fixed baselines
4. `run_visuals`: Trains all configs, saves side-by-side image grids
5. `run_vram_profile`: Uses `torch.cuda.max_memory_allocated()` to measure peak GPU memory

### `experiments/run_theory.py` (380 lines)

**3 theory experiments**:
1. `measure_gradient_variance`: At a fixed model state, samples 30 different mini-batches, computes gradient for each, measures std/mean of gradient norms
2. `run_lr_scaling`: Tests 4 LR rules (fixed, linear, sqrt, inverse_sqrt) at each batch size
3. `run_difficulty_analysis`: Computes edge density + HF energy ratio for each image, correlates with batch-size sensitivity

### `experiments/run_seeds_and_jpeg.py` (240 lines)

1. `run_jpeg_comparison`: Compresses each image with PIL JPEG at 13 quality levels, measures PSNR and file size, creates rate-distortion curve
2. `run_multi_seed`: Runs SIREN sweep 3 times with different `torch.manual_seed()`, decomposes variance into inter-image vs inter-seed

---

## Part 4: Key Concepts Cheat Sheet

| Term | Simple Explanation |
|------|-------------------|
| **INR** | Store image as a function (neural network) instead of a pixel grid |
| **SIREN** | Neural network with sin() activations — good at learning fine details |
| **COIN** | Neural network with ReLU + positional encoding — alternative approach |
| **Batch size** | How many pixels the network sees per training step |
| **Full batch** | See ALL pixels every step (exact gradient, slow) |
| **Mini-batch** | See a random subset (noisy gradient, fast) |
| **PSNR** | Quality metric in dB. 10*log10(1/MSE). Higher = better |
| **SSIM** | Perceptual similarity. 0 to 1. Higher = better |
| **Spectral bias** | Neural networks learn low frequencies first, high frequencies last |
| **FFT** | Converts image from pixel space to frequency space |
| **Gradient variance** | How much the training signal changes between different mini-batches |
| **Iso-time** | Compare methods at the same wall-clock time (fair comparison) |
| **Adaptive schedule** | Change batch size during training (small→large) |
| **Rate-distortion** | Trade-off between file size and image quality |

---

## Part 5: How to Run Everything

```bash
# Install dependencies
pip install -r requirements.txt

# Download Kodak dataset
python data/download_kodak.py

# Run main SIREN sweep
python -m experiments.run_sweep

# Run extended experiments (COIN, iso-time, adaptive, visuals, VRAM)
python -m experiments.run_extended --experiment all

# Run theory experiments (gradient variance, LR scaling, difficulty)
python -m experiments.run_theory --experiment all

# Run JPEG comparison + multi-seed
python -m experiments.run_seeds_and_jpeg --experiment all

# Analyze results
python -m experiments.analyze_results
python -m experiments.analyze_extended
```
