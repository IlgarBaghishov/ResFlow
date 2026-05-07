# ResFlow

3D generative models for siliciclastic-reservoir simulation. Generates geologically plausible binary-facies volumes conditioned on layer-type, geological scalars, and well observations — with multi-block parallel denoising that assembles arbitrarily large reservoirs from a fixed 64×64×32 base model.

## Highlights

- **Eight reservoir architectures from one model.** A single Flow-Matching network handles channel-belt jigsaw, channel-belt labyrinth, meander-oxbow, point-bar shoestring, sheet-distal, sheet-proximal, lobe, and delta layer types via a layer-type one-hot in the conditioning vector.
- **Well-conditioned inpainting.** The same weights produce unconditional samples (empty mask) or well-conditioned samples (1–N vertical/L-shaped wells) — no separate "inpainting" model. Hard replacement at the final denoising step guarantees exact agreement at known voxels.
- **Big-reservoir multi-block assembly.** Parallel block denoising with overlap blending lets you generate reservoirs of arbitrary plan size (e.g. 600×600×32) by tiling 64×64×32 blocks with hard or soft transitions between layer types. See `resflow/assembly/big_reservoir_multi.py`.
- **Round-trip property evaluation.** A separately-trained CNN3D property predictor scores whether generated samples actually preserve the requested NTG, geometry, and azimuth.
- **Method comparison.** Diffusion (DDPM/DDIM), Flow Matching, MeanFlow, and Rectified Flow share an apples-to-apples training loop on a smaller geological lobe benchmark. Flow Matching was chosen for the reservoir scaling work because it gave the best tradeoff between sample quality, NFE, and CFG stability — see "Methods" below.

## Quick start

```bash
pip install -e .
```

End-to-end on the SiliciclasticReservoirs dataset (`AnonymouScientist/SiliciclasticReservoirs`):

```bash
cd examples/reservoirs/inpainting

# Train the well-conditioned inpainting model (auto-downloads dataset to $SCRATCH on first run)
python train.py                        # single GPU
sbatch run_A100.sh                     # 4 nodes × 3 A100s on a 3-A100-per-node HPC
sbatch run_GH200.sh                    # 8 nodes × 1 GH200 on a GH200 HPC

# Sample one cube per layer type, with and without 5-well "+"-pattern conditioning
python sample_30min_demo.py

# Build a 10×10 big reservoir of, say, lobe blocks with mixed scalars
cd ../big_reservoir/lobes
python generate.py
python visualize.py
```

## Datasets

| Dataset | Volumes | Conditioning | Role |
|---|---|---|---|
| **SiliciclasticReservoirs** | 1M binary 64×64×32 cubes across 8 layer-type families | layer-type one-hot + 4 universal scalars (NTG, width, depth, azimuth) + 5 family-specific scalars | **Headline.** Used for inpainting + big-reservoir assembly. |
| **Lobes** | ~89k binary 50×50×50 single-lobe cubes | 5 continuous scalars (height, radius, aspect ratio, angle, NTG) | Smaller geological benchmark used during method comparison. |

## What's in here

```
resflow/
├── models/
│   ├── unet3d.py            # 3D UNet, continuous-cond + learned null embedding for CFG
│   └── cnn3d.py             # 3D CNN property predictor (round-trip evaluation)
├── methods/
│   ├── diffusion.py         # DDPM + DDIM sampling, x0-clipping for stable CFG
│   ├── flow_matching.py     # OT Flow Matching, Euler integration  ← used for reservoirs
│   ├── meanflow.py          # MeanFlow with JVP-based training, 1-step capable
│   └── rectified_flow.py    # 2-Rectified Flow with forward/backward/bidirectional reflow
├── utils/
│   ├── data_reservoirs.py   # Sharded loader for SiliciclasticReservoirs
│   ├── data_lobes.py        # Lobe dataset + on-the-fly inpaint mask wrapper
│   ├── masking.py           # Wells / boundaries / cross-sections (3D inpainting)
│   ├── training.py          # AdamW + cosine LR + EMA + train_model_inpaint
│   └── plotting{,_lobes}.py # Cross-section grids, wells overlay, loss curves
└── assembly/
    └── big_reservoir_multi.py  # Multi-block parallel denoising with overlap blending

examples/
├── reservoirs/                # ← centerpiece
│   ├── inpainting/            #   training, sampling, eval-loss, 30-min smoke run
│   └── big_reservoir/         #   per-layer-type uniform/long generators + 3-type sequence
├── lobes/                     # smaller geological benchmark
│   ├── standard/              #   unconditional + class-CFG generation
│   ├── inpainting/            #   wells / boundaries / cross-sections inpainting
│   ├── CNN/                   #   CNN3D property predictor train/evaluate
│   └── big_reservoir/         #   lobe-only multi-block assembly (precursor to reservoirs/)
```

## Reservoir details

### Conditioning vector (18-D)

```
[ layer_one_hot (8) | NTG | width_cells | depth_cells | sin(az) | cos(az) |
  asp | mCHsinu | mFFCHprop | probAvulInside | trunk_length_fraction ]
```

The five family-specific scalars are zeroed for layer types that don't use them. Universal scalars are normalized to the global per-feature min/max from the training cond cache. Azimuth is encoded as `(sin, cos)` for 360° periodicity.

### Inpainting via channel concatenation

The 3D UNet takes 3 input channels — `[noisy_x, known_data, mask]` — and outputs 1 channel. Inpaint context is set on the model via stateful `set_inpaint_context(mask, data)` / `clear_inpaint_context()`, so methods (`FlowMatching`, `Diffusion`, …) stay completely inpainting-unaware. Mask convention: `1 = known (keep)`, `0 = unknown (generate)`.

Training distribution mixes unconditional and well-conditioned samples (30% empty mask, 70% with 1–5 wells), so a single set of weights serves both modes at sampling time.

### Big-reservoir multi-block assembly

`generate_big_reservoir_multi` tiles a grid of `BlockSpec`s across X×Y, where each block has its own layer type and per-block scalars. Adjacent blocks share an overlap region (`overlap_xy ∈ {12, 16, 24}` in our experiments) — denoising is run jointly across all blocks in parallel, with the overlapping noise updated as a smooth blend of the contributing blocks at every step. Two transition modes:

- **Hard**: each block sees only its own conditioning everywhere, blending happens only in the noise update.
- **Soft**: blocks linearly interpolate cond vectors across overlaps, producing smoother facies transitions at the cost of a small in-distribution drift inside the overlap.

## Methods (comparison summary)

| Method | What it learns | Sampling | NFE / sample | Notes |
|---|---|---|---|---|
| **Diffusion** (DDPM/DDIM) | Noise prediction | Iterative denoising | ~50 | Linear β-schedule; DDIM clips x0 + recomputes ε for CFG stability |
| **Flow Matching** ✅ | Velocity field | Euler ODE | ~50 | Chosen for reservoirs |
| **MeanFlow** | Mean velocity | Single-step capable | 1 (embedded CFG) or 2/step | JVP target, EMA target network |
| **Rectified Flow** | Straightened velocity | Euler ODE | <50 | Forward / backward / bidirectional reflow on coupled pairs |

We picked Flow Matching for the reservoir work because it gave clean, stable samples at ~50 steps, integrated with channel-concat inpainting without any architectural changes, and produced no visible CFG drift even at high guidance scales.

## Architecture: model ↔ method decoupling

Methods and models communicate through a minimal interface:

```python
model(x, t, cond)                # conditional forward pass
model(x, t)                      # unconditional (model uses its own null representation)
model(x, t, cond, drop_mask=m)   # mixed batch for training-time CFG
```

Methods decide *when* to drop conditioning and *how* to combine cond/uncond at sampling time. Models decide *what* "unconditional" means internally (learned null embedding, null class token, …). Any method works with any model — including the inpainting variant, since inpaint context lives on the model, not the method.

## Reproducing the reservoir results

```bash
# 1. Train the FM-inpaint model on 1M cubes
cd examples/reservoirs/inpainting
sbatch run_A100.sh           # or run_GH200.sh

# 2. Sample 8-layer-type demo (one cube per type, with and without 5 wells)
python sample_30min_demo.py

# 3. Evaluate held-out FM loss on train/val/test
sbatch run_eval.sh

# 4. Build big reservoirs by layer family
cd ../big_reservoir
python setup_uniform.py             # regenerate per-family generate.py templates
python long_setup.py                # 1×10 long variants
cd lobes && python generate.py && python visualize.py

# 5. Train CNN3D property predictor and run round-trip eval
cd ../../lobes/CNN
python train.py
python evaluate.py
```

Checkpoints land in `$SCRATCH/genflows_runs/...` by default; override with `RESERVOIR_DATA_DIR` and the `--ckpt` flags shown in each script.

## Citation

```
(submitted; under review — anonymous for double-blind submission)
```

## License

MIT
