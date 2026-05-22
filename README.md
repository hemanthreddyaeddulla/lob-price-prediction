# LOB Price Prediction - Wunder Predictorium Competition

**Ranked 53 out of 4,900+ participants (Top 1.1%). Final weighted Pearson correlation: 0.3083.**

This is my full solution to the Wunder Predictorium, a quantitative-finance competition where you predict two future price-movement indicators from streaming Limit Order Book (LOB) data, one snapshot at a time, on a single CPU core, with a 60-minute budget for roughly 1.5 million prediction steps. The setup mirrors what a real low-latency market-making system has to do: implicit signal estimation from order-book state, online, with no lookahead.

I built a 10-model GRU ensemble with a Conv1D cross-feature mixer, trained on 158 engineered features I derived from the 32 raw LOB inputs (all computable in O(1) per step), and deployed it through stateful ONNX Runtime so the ensemble fits comfortably inside the 2.4 ms/step latency budget.

| Final result | |
|---|---|
| Rank | 53 / 4,900+ (Top 1.1%) |
| My score | 0.3083 |
| 1st place | 0.3537 (gap ~14.7%) |
| Local CV score | 0.3762 |
| Inference time | ~1.6 ms/step on 1 vCPU (budget: 2.4 ms) |
| Models in ensemble | 10 (5 folds x 2 random seeds) |
| Features per step | 158 (engineered from 32 raw) |

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Why It Is Hard](#2-why-it-is-hard)
3. [Data Exploration I Did First](#3-data-exploration-i-did-first)
4. [The Central Insight That Drove Everything](#4-the-central-insight-that-drove-everything)
5. [Feature Engineering: 32 to 158, All O(1)](#5-feature-engineering-32-to-158-all-o1)
6. [Model Architecture](#6-model-architecture)
7. [Loss Function and Training Schedule](#7-loss-function-and-training-schedule)
8. [Cross-Validation Strategy](#8-cross-validation-strategy)
9. [Ensemble Strategy](#9-ensemble-strategy)
10. [Inference: Stateful ONNX](#10-inference-stateful-onnx)
11. [Things I Tried That Did Not Work](#11-things-i-tried-that-did-not-work)
12. [Results Breakdown](#12-results-breakdown)
13. [Every File in This Repo](#13-every-file-in-this-repo)
14. [How To Reproduce](#14-how-to-reproduce)
15. [Tech Stack](#15-tech-stack)
16. [Lessons Learned](#16-lessons-learned)
17. [If I Had Another Month](#17-if-i-had-another-month)
18. [Contact](#18-contact)

---

## 1. The Problem

Each input sequence is 1,000 time steps of 32 anonymized LOB features:

- `p0..p5` - the six bid price levels
- `p6..p11` - the six ask price levels
- `v0..v5` - the six bid volumes
- `v6..v11` - the six ask volumes
- `dp0..dp3` - four trade price features
- `dv0..dv3` - four trade volume features

For every step I receive a `DataPoint` with `(seq_ix, step_in_seq, need_prediction, state)` and must return two predictions: `t0` and `t1`, both anonymized future price-movement indicators. Steps 0 through 98 of each sequence are warmup (predictions not scored). Steps 99 through 999 are scored. When `seq_ix` changes I have to reset every piece of internal state (feature engine memory, RNN hidden states) because sequences are independent.

The scoring metric is **Weighted Pearson Correlation** with weights equal to `|y_true|`. Predictions are clipped to `[-6, 6]` before scoring. The final score is the mean of WPC(t0) and WPC(t1). Because of the absolute-value weighting, a single big move (target around 5) carries roughly 50x the influence of a tiny move (target around 0.1). The metric is rank-insensitive to constant scale or offset, but cares a lot about getting the direction and relative magnitude of large moves right.

Constraints:

- 1 vCPU, no GPU, 16 GB RAM
- 60 minutes total for ~1,500 sequences of 1,000 steps
- No internet, no external data
- Python 3.11 in a `python:3.11-slim-bookworm` Docker base
- 5 submissions per day

The latency budget works out to **2.4 ms per step** for the entire pipeline: feature engineering plus model inference.

## 2. Why It Is Hard

Three things make this competition hard, and all three shaped my design:

**(1) Individual features are weak.** I computed every feature-target correlation across the dataset. The strongest single feature on t0 is `p10` at 0.133. The strongest on t1 is `v8` at 0.035. There is no single number on a single step that tells you much. The signal lives in *temporal patterns* and *combinations* of features.

**(2) The targets are highly autocorrelated but invisible.** t0 has lag-1 autocorrelation 0.749. t1 has lag-1 autocorrelation 0.966. If I could see the previous target, an oracle baseline gets WPC 0.68 on t0 and 0.91 on t1. But the targets are not available at inference. The model has to implicitly reconstruct a latent signal from feature history using its hidden state.

**(3) Strict streaming constraints.** I cannot keep a growing buffer of history. I cannot batch sequences. I cannot use a GPU. Every per-step cost has to fit under ~2.4 ms.

## 3. Data Exploration I Did First

Before writing any model code, I spent time understanding the data. Findings that drove the design:

| What I checked | What I found |
|---|---|
| t0, t1 distributions | Heavy-tailed, roughly zero-mean, t0 wider than t1 |
| Per-feature correlation with each target | Max 0.133 (t0), max 0.035 (t1). Weak across the board. |
| Feature-target lag correlations | `p10` has 0.44 correlation with *lagged* t0. It partially encodes the previous target. |
| Target autocorrelation at lags 1, 2, 5, 10, 20, 50 | t0 decays fast, t1 holds 0.71 even at lag 10 |
| Oracle WPC if I could see previous target | t0: 0.68, t1: 0.91. This number framed the entire project. |
| Train/valid distribution match | Looked fine on histograms. Distribution shift to leaderboard was a separate later issue. |
| Sequence boundaries | Sequences are independent (confirmed by checking that boundary deltas behave normally) |

The most important number on this list is the oracle WPC: it told me the ceiling was high if a model could reconstruct the latent target signal from feature history.

## 4. The Central Insight That Drove Everything

The autocorrelation finding is the spine of the whole solution. Restated:

> The target is highly autocorrelated, so a perfect predictor for it already exists - the previous target itself. But I cannot see the previous target. So the entire job reduces to: **how well can a model implicitly track the latent target signal from the feature stream using its hidden state?**

This single observation drove every downstream decision:

- **Recurrent architecture (GRU)** - the hidden state is the implicit tracker.
- **Stateful inference** - if I reprocess history every step the model starts cold and the hidden state has no time to learn.
- **Multi-scale EMA features** - I want the input stream to carry usable summaries of recent history at multiple timescales (so the GRU can latch onto the slow component for t1 and the fast component for t0).
- **Separate prediction heads** - t0 and t1 live on different timescales; let each head specialize on top of the shared encoder.
- **Two-phase loss** - first stabilize with MSE, then nudge toward the actual metric.
- **Ensembling** - reduce variance in the latent tracker by averaging many independent estimators.

If anything in the design surprises someone, it usually traces back to this one observation.

## 5. Feature Engineering: 32 to 158, All O(1)

Implemented in `src/features.py` (training) and mirrored exactly in `solution/feature_eng.py` (inference). Every feature has constant time and constant memory cost per step. No rolling windows, no growing buffers. The persistent state per sequence is **78 float32 values, about 312 bytes**.

| Group | Count | Description |
|---|---|---|
| Raw features | 32 | The original LOB state passed through |
| Step-to-step deltas | 32 | `feature[t] - feature[t-1]`. Rate of change. |
| Fast EMA (alpha=0.3) | 10 | ~3-step memory, on the 10 most informative raw features |
| Slow EMA (alpha=0.01) | 10 | ~100-step memory |
| Glacial EMA (alpha=0.003) | 10 | ~333-step memory, designed for the very persistent t1 |
| Slow EMA deviation | 10 | `feature - slow_ema`. Mean-reversion signal. |
| Glacial EMA deviation | 10 | `feature - glacial_ema`. Long-horizon mean-reversion. |
| Momentum (EMA of deltas) | 10 | Smoothed rate-of-change |
| Volume imbalance | 6 | `(bid_vol - ask_vol) / (|bid_vol| + |ask_vol|)` per level |
| Bid-ask spread | 6 | `ask_price - bid_price` per level |
| Order Flow Imbalance | 6 | `delta_bid_vol - delta_ask_vol`. Net new liquidity. |
| Cumulative OFI (EMA of OFI) | 6 | Persistent order flow direction |
| Microprice proxy + pressure | 2 | Volume-weighted blend of best bid and best ask, plus the bid-side fraction of best-level volume |
| Cross-features | 8 | Handpicked products like `p10*v0`, `mean(bid_prices)*mean(bid_volumes)`, `spread*vol_imb`, scaled down by 0.1 so they do not dominate |
| **Total** | **158** | |

The 10 features picked for the EMA pipeline came from a gradient-boosting importance scan: `p10, p9, p2, p5, dp0, v8, v0, v2, dp3, p8`. `p10` is by far the most informative (it has that 0.44 correlation with lagged t0).

**Three EMA timescales, not one.** This is deliberate. t0 lives on a fast timescale (decorrelates within ~5 steps) so the fast EMA matters most for it. t1 lives on a much slower timescale (0.71 autocorrelation even at lag 10) so the glacial EMA matters most for it. The slow EMA in between catches medium-horizon trend. Including all three lets one model handle both targets well.

## 6. Model Architecture

Implemented in `src/models.py`. One forward pass per step:

```
Input (158 features, per step)
   |
[LayerNorm]
   |
[Conv1D Cross-Feature Mixer]  (3 pointwise layers, GELU)
   |     (residual)
   +----<-----+
   |
[2-layer GRU, hidden=128]
   |
   |---> Hidden state (256 floats, persists across steps)
   |
   +---> [t0 MLP head: 128 -> 64 -> 1]
   |
   +---> [t1 MLP head: 128 -> 64 -> 1]
```

**Why a Conv1D mixer?** LOB features have structure that I do not want to hand-engineer combination by combination. The Conv1D mixer is just pointwise convolutions across the feature dimension (kernel size 1). It learns interactions like "bid level 0 times ask level 0 volume" without me writing those crosses out by hand. The residual connection means the model can choose to ignore the mixer if it wants. In practice ablations showed the mixer helps a modest but consistent amount.

**Why GRU instead of LSTM or Transformer?** I trained both. LSTM is the closest competitor; per-fold scores are essentially tied with GRU but GRU runs faster on CPU. Transformer-style attention was infeasible at the latency budget without dramatic context truncation. State-space models (Mamba/S4) would have been the right next experiment if I had another month.

**Why hidden size 128?** Tried 64, 128, 192, 256. Above 128 the per-model score plateaus while training and inference both get more expensive. Below 128 the model is noticeably weaker. 128 is the elbow.

**Why two heads?** t0 and t1 have such different dynamics that giving each its own decoder lets the model specialize while sharing the temporal encoder. Tiny addition in parameters, small but real gain.

## 7. Loss Function and Training Schedule

Implemented in `src/losses.py`. Two-phase, with a smooth transition.

**Phase 1 (first 85% of epochs): plain MSE on clipped predictions.** Vanilla MSE is stable. Gradients behave. The model converges on the easy structure first.

**Phase 2 (final 15% of epochs): blend in differentiable Weighted Pearson Correlation.** Ramps `pearson_weight` linearly from 0 up to 0.5 over 10 epochs, with `mse_weight` going from 1.0 down to 0.5 in lockstep so the total loss magnitude is sensible. The Pearson term is the actual competition metric, made differentiable.

This sequencing matters. The natural instinct is to train with a metric-aligned loss from step one. I tried that. Specifically I tried `|target|`-weighted MSE, which matches the metric weighting directly. It overfit to extreme values badly. Plain MSE then Pearson fine-tune was strictly better in every fold.

**Other training settings that ended up sticking:**

- AdamW optimizer
- Learning rate 5e-4 with cosine annealing to 1e-6
- Weight decay 5e-4
- Batch size 64 sequences (each sequence is 1000 steps, processed in full)
- Max 40 epochs, early stopping with patience 15 on validation WPC
- Gradient clipping at 1.0 max norm
- Feature dropout: per-sequence, drop ~5% of feature channels at random
- Feature scaling: per-sample, per-feature random multiplier in [0.9, 1.1]
- Gaussian feature noise with sigma 0.01

The augmentation block (dropout + scale + noise) gave a small but reliable regularization boost. Removing it shifted scores down by about 0.005 average WPC.

## 8. Cross-Validation Strategy

Implemented in `src/dataset.py` (`split_sequences_kfold`).

I combined `train.parquet` (10,721 sequences) and `valid.parquet` (1,444 sequences) into a single pool of 12,165 sequences, then ran 5-fold cross-validation split by sequence ID. The fold seed is held constant (42) across all training seeds so the fold boundaries are deterministic.

**Why this was a big deal.** With a single random train/valid split I was getting ~0.27 WPC on validation. Switching to proper 5-fold CV produced 0.29+ per fold consistently and gave me much more reliable model selection. The variance from a single split was big enough to mislead which hyperparameter settings looked best. After switching, my decision quality went up across the board.

This sounds obvious in hindsight. With 12K sequences, a single split is just not enough resolution.

## 9. Ensemble Strategy

Implemented in `src/train_ensemble.py` (training orchestration) and `solution/solution.py` (inference-time averaging).

The final submission is **10 GRU models**: 5 folds x 2 random seeds (42 and 123), all the same architecture (hidden=128, 2 layers, mixer 3 layers, head 64). At inference time I run all 10 models on every step and average their predictions.

**Progression I measured:**

| Configuration | Avg WPC |
|---|---|
| Single best model | ~0.29 |
| 5-model ensemble (1 seed x 5 folds) | 0.366 |
| 10-model ensemble (2 seeds x 5 folds) | 0.376 (local CV) |

That ~0.08 gap from single-model to 10-model ensemble is the single biggest lever in the entire project. Larger than any architecture change, larger than any feature change, larger than any loss change. Ensemble diversity wins.

**Equal weights vs learned weights.** I implemented out-of-fold weight optimization in `src/oof_validate.py` using Nelder-Mead at the config-group level (not per-model, to keep the search dim small and prevent overfitting the weights). The optimized weights were within ~0.001 WPC of equal weights. I shipped equal weights.

## 10. Inference: Stateful ONNX

Implemented in `src/export_onnx.py` (export) and `solution/solution.py` (inference).

I export each trained PyTorch model to ONNX with **hidden state as an explicit input and output**:

```
ONNX inputs:   features (1, 158)   +   hidden (2, 1, 128)
ONNX outputs:  prediction (1, 2)   +   new_hidden (2, 1, 128)
```

At inference time the submission keeps the hidden state on the Python side as a numpy array. Each step the ONNX model sees only the current snapshot and the previous hidden state, then returns a new prediction and a new hidden state. No history is reprocessed. This is the key to the latency budget.

**Crucial detail: I run all 10 models through the 99-step warmup, not just from step 99.** If I started cold at step 99 the hidden states would have no context. By running through warmup steps even though I do not return predictions, the hidden states accumulate the temporal pattern they will need.

**Per-step timing:**

| Stage | Time |
|---|---|
| Feature engineering (32 -> 158) | ~0.05 ms |
| 10 ONNX models | ~2.5 ms total (~0.25 ms each) |
| Averaging | ~0.01 ms |
| Total | ~1.6 ms (well under 2.4 ms budget) |

ORT session options I tuned: `intra_op_num_threads=1` and `inter_op_num_threads=1`. ONNX Runtime would otherwise try to grab all cores and oversubscribe on the 1-vCPU evaluation box, which paradoxically slows everything down.

## 11. Things I Tried That Did Not Work

Negative results, with one-line reasons. Each of these took at least a full training cycle to confirm.

| Idea | Outcome | Why |
|---|---|---|
| LSTM in place of GRU | Tied per-fold, GRU is faster | Architectures equivalent on this data |
| Larger hidden size (192, 256) | Same per-model score as 128 | Capacity not the bottleneck |
| Upweighting t1 in the loss | Hurt overall score | t1 signal is too weak; pushing harder amplified noise |
| Target-weighted MSE from epoch 0 | Overfit to extremes | Metric-aligned loss is unstable early |
| Autoregressive target feeding (use last prediction as feature) | Catastrophic | Error compounds along the sequence |
| Larger feature dropout (10%, 20%) | Hurt | Too aggressive given how weak individual features are |
| Per-target separate models (one GRU for t0, another for t1) | Slight regression | Shared encoder is the right call |
| Learned per-model ensemble weights | Within 0.001 of equal | Equal weighting is fine |
| Conv-only (no recurrence) | Big regression | The hidden state is essential for the latent target signal |
| Removing Conv1D mixer | Small regression | Mixer is worth its cost |

A short list of negative results is often more useful to a recruiter than a long list of features, because it tells you what the engineer actually had to fight through.

## 12. Results Breakdown

**Per-target performance on local cross-validation:**

| Target | Local WPC | Oracle WPC | % of oracle |
|---|---|---|---|
| t0 | 0.479 | 0.68 | 70% |
| t1 | 0.274 | 0.91 | 30% |

t0 is well-tracked. The model captures most of what is achievable given the autocorrelation ceiling.

t1 is the obvious bottleneck. Its lag-1 autocorrelation of 0.966 combined with the extremely weak per-step feature signal (max 0.035 single-feature correlation) makes the latent signal very slow and very hard to estimate from features. Even a perfect tracker would still be far below the oracle. With more time I would have focused entirely on long-memory architectures for t1.

**Local CV vs leaderboard gap.** My local 5-fold CV reported 0.3762 average WPC. The leaderboard gave me 0.3083. The gap is real and reflects distribution shift between validation data and the held-out test set, which is a familiar issue in financial ML. Post-hoc calibration (a single scale or affine transform fit on validation and applied at inference) is the standard fix and I would try it if iterating further.

**Final standing:** 53 / 4,900+. Top 1.1%. 1st place finished at 0.3537, prize cutoff (8th) at 0.3307. Gap to 1st is ~14.7%.

## 13. Every File in This Repo

```
.
|-- README.md                           This file. Complete project overview.
|-- utils.py                            Official scoring code: DataPoint class and
|                                       ScorerStepByStep. Provided by the competition.
|                                       I did not modify it because the scoring system
|                                       uses it directly.
|
|-- docs/
|   `-- Methodology_WunderPredictorium.md  Formal methodology write-up. Companion to
|                                          this README, in a more academic voice.
|
|-- src/                                Training pipeline (NOT included in the
|   |                                   submission zip - this is the offline side).
|   |-- features.py                     The OnlineFeatureEngine. Maps 32 raw features
|   |                                   to 158 engineered features in O(1) per step.
|   |                                   Has both a per-step `process()` and a batch
|   |                                   `process_sequence()` for training prep.
|   |-- models.py                       GRUModel, LSTMModel, CrossFeatureMixer,
|   |                                   PredictionHead. Each model has both a
|   |                                   full-sequence forward (for training) and a
|   |                                   forward_step (for ONNX export).
|   |-- losses.py                       plain_mse_loss, soft_weighted_mse_loss,
|   |                                   huber_loss, weighted_pearson_loss (the
|   |                                   differentiable metric), per_target_mse,
|   |                                   combined_loss (the two-phase blend used in
|   |                                   training).
|   |-- dataset.py                      LOBDataset (PyTorch Dataset over per-sequence
|   |                                   pre-computed features), load_sequences (parquet
|   |                                   reader and grouper), split_sequences_kfold
|   |                                   (deterministic k-fold split by sequence ID),
|   |                                   create_dataloaders (the top-level helper used
|   |                                   by train.py).
|   |-- train.py                        Single-model training entry point. Implements
|   |                                   the two-phase loss schedule, cosine LR,
|   |                                   feature augmentation, early stopping, best
|   |                                   checkpoint tracking. Drives one fold of one
|   |                                   seed.
|   |-- train_ensemble.py               Orchestrates many train.py runs. Loops over
|   |                                   configs x seeds x folds, then exports each
|   |                                   best checkpoint to ONNX. This is what built
|   |                                   the 10-model ensemble.
|   |-- export_onnx.py                  Wraps a trained model in a stateful single-step
|   |                                   forward, exports to ONNX with hidden state as
|   |                                   explicit I/O, and verifies the ONNX output
|   |                                   matches the PyTorch output to numerical
|   |                                   tolerance.
|   |-- oof_validate.py                 Honest out-of-fold ensemble evaluation. For
|   |                                   each fold model, only scores sequences in its
|   |                                   own validation set. Optional weight optimization
|   |                                   via Nelder-Mead at the config-group level.
|   |-- validate.py                     End-to-end local validation. Runs solution/
|   |                                   against valid.parquet using the official
|   |                                   ScorerStepByStep. Reports per-target WPC and
|   |                                   extrapolated test-set timing.
|   `-- __init__.py                     Empty. Makes src/ importable.
|
`-- solution/                           The actual submission package. This is what
    |                                   gets zipped and uploaded.
    |-- solution.py                     Defines class PredictionModel with a predict()
    |                                   method matching the competition contract.
    |                                   Loads all .onnx files in this directory,
    |                                   runs them all on every step (including warmup,
    |                                   to keep hidden states warm), and averages
    |                                   predictions. Resets all state on a new seq_ix.
    |-- feature_eng.py                  Self-contained copy of OnlineFeatureEngine
    |                                   (numpy-only, no torch dependency) so the
    |                                   submission zip has no dependency on src/.
    `-- *.onnx (10 files)               The 10 trained ensemble models that produced
                                        the 0.3083 leaderboard score. Each is roughly
                                        1.2 MB. Naming pattern:
                                        gru_h128_l2_s{seed}_fold{N}.onnx
```

**Files excluded from the repo (size or licensing):**

- `datasets/train.parquet` (1.5 GB) and `datasets/valid.parquet` (225 MB): competition data, not redistributable.
- `checkpoints/` (424 MB): all the intermediate `.pt` files from training. The relevant ones are already exported to ONNX in `solution/`.
- Submission zip and Python cache directories.

## 14. How To Reproduce

```bash
# Python 3.11
pip install numpy pandas torch onnx onnxruntime tqdm pyarrow scipy

# 1. Train the 10-model ensemble (5 folds x 2 seeds, ~hours on CPU)
python src/train_ensemble.py --configs gru_128 --seeds 42 123

# 2. Export every best checkpoint to stateful ONNX
#    (train_ensemble.py does this automatically; you can also run manually:)
python src/export_onnx.py --checkpoint checkpoints/gru_h128_l2_s42_fold0_best.pt --verify

# 3. Optional: run honest out-of-fold validation across the ensemble
python src/oof_validate.py --checkpoint-dir checkpoints/ --seeds 42 123

# 4. Local end-to-end validation against valid.parquet using the official scorer
python src/validate.py

# 5. Package the submission
cd solution && zip -r ../submission.zip . && cd ..
```

The included `.onnx` files in `solution/` are the actual models from the 0.3083 leaderboard run, so step 4 works directly out of a fresh clone without retraining.

## 15. Tech Stack

- **Modeling:** PyTorch 2.x for training, ONNX Runtime for CPU inference
- **Data:** NumPy, Pandas, PyArrow (Parquet)
- **Optimization:** SciPy (Nelder-Mead for ensemble-weight search)
- **Hard constraints:** 1 vCPU, no GPU at inference, Python 3.11

## 16. Lessons Learned

A few things that crystallized for me on this project:

1. **Ensemble discipline beats architectural novelty.** A boring average of 10 identical-architecture GRUs trained on different folds and seeds outperformed every fancier architecture I tried.
2. **Understand the metric before designing the loss.** Training against a metric-aligned loss is not always the right move. The indirect path (plain MSE first, Pearson fine-tune at the end) won here.
3. **Recurrent hidden state is a powerful implicit tracker.** When the target is highly autocorrelated but unobservable, a well-trained RNN learns to approximate the latent signal. This beat everything that tried to reconstruct the target explicitly.
4. **K-fold CV is not optional in small-data regimes.** A single train/valid split misled me by ~0.02 WPC, which in this scoring is the difference between a useful model and a useless one.
5. **O(1) feature engineering is what keeps ensembling viable.** Constant-time multi-scale EMAs and OFI cost essentially nothing per step, so the model gets richer inputs without breaking the latency budget.
6. **Negative results are signal too.** Every failed experiment I documented (autoregressive feeding, weighted MSE, separate-target models) sharpened the design.

## 17. If I Had Another Month

Concrete next steps I would try, ranked by expected impact:

- **State-space models (Mamba/S4) for t1.** The t1 bottleneck is long-memory tracking. SSMs are built for exactly this.
- **Post-hoc calibration to close the local-to-leaderboard gap.** A single scale/affine on each target, fit on held-out validation. Cheap to try, plausibly worth 0.01-0.02 WPC.
- **Architecture diversity in the ensemble.** Right now all 10 models are the same architecture. Mixing GRU with a Mamba variant should add real ensemble diversity.
- **Attention over a fixed-size hidden-state window.** Lets the model adaptively weight recent steps without breaking the streaming constraint.
- **Target reconstruction as an auxiliary task.** Predict not just t0/t1 but a smoothed version of recent targets too, multi-task. Might give the encoder a stronger learning signal.

## 18. Contact

Hemanth Reddy Aeddulla. Open to ML, quant research, and quantitative engineering roles. Reach me via my [GitHub profile](https://github.com/hemanthreddyaeddulla).
