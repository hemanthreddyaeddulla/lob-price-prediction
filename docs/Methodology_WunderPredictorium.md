# Real-Time Price Prediction from Limit Order Book Data

## Technical Methodology | Wunder Predictorium Competition

**Result: Rank 53 / 4,900+ participants (Top 1.1%) | Final Score: 0.3083**

---

## 1. Executive Summary

This document describes the methodology used to achieve a top 1.1% ranking in the Wunder Predictorium, a quantitative finance competition requiring real-time prediction of future price movement indicators from Limit Order Book (LOB) data.

The core challenge was building a system that could process streaming market data one snapshot at a time, under strict latency and resource constraints - mirroring the requirements of production trading systems.

**Final Solution:**

| Component | Detail |
|-----------|--------|
| Architecture | 10-model GRU ensemble with Conv1D cross-feature mixing |
| Features | 158 engineered features from 32 raw inputs (all O(1) per step) |
| Inference | Stateful ONNX Runtime, ~1.6ms per step (budget: 2.4ms) |
| Training | 5-fold CV, two-phase loss (MSE then Pearson), AdamW with cosine decay |
| Final Score | 0.3083 weighted Pearson correlation (1st place: 0.3537) |

---

## 2. Problem Definition

### Task

Predict two anonymized future price movement indicators (t0, t1) from a stream of 32 Limit Order Book features, delivered one step at a time.

### Data

| Property | Value |
|----------|-------|
| Training sequences | 10,721 (+ 1,444 validation) |
| Steps per sequence | 1,000 (steps 0-98 warmup, 99-999 scored) |
| Features per step | 32 (6 bid prices, 6 ask prices, 6 bid volumes, 6 ask volumes, 4 trade prices, 4 trade volumes) |
| Targets | t0 (fast-decaying indicator), t1 (persistent indicator) |
| Feature normalization | Pre-normalized to approximately [-5.2, 5.2] |

### Constraints

| Resource | Limit |
|----------|-------|
| Compute | 1 vCPU core (no GPU) |
| Memory | 16 GB RAM |
| Time | 60 minutes for ~1.5M prediction steps |
| Latency budget | ~2.4 ms per step |
| Environment | Offline (no internet), Python 3.11 |

### Evaluation Metric

Weighted Pearson Correlation Coefficient, where each sample is weighted by |y_true|. Predictions are clipped to [-6, 6]. The final score averages the correlation across both targets.

This metric heavily rewards correct predictions on large price movements - a sample with |target| = 5 carries 50x the influence of one with |target| = 0.1.

---

## 3. Data Analysis and Key Insights

### 3.1 Target Autocorrelation - The Central Finding

The most important discovery was extreme temporal persistence in the targets:

| Lag | t0 Autocorrelation | t1 Autocorrelation |
|-----|--------------------|--------------------|
| 1 | 0.749 | 0.966 |
| 2 | 0.636 | 0.937 |
| 5 | 0.414 | 0.851 |
| 10 | 0.232 | 0.710 |
| 20 | 0.089 | 0.447 |
| 50 | -0.007 | -0.059 |

An oracle with access to the previous step's target achieves 0.68 WPC on t0 and 0.91 on t1. However, previous targets are unavailable at inference time. The model must implicitly reconstruct and track these latent signals through its recurrent hidden state.

This is fundamentally the same challenge as latent signal estimation in real market-making systems.

### 3.2 Weak Individual Feature Signals

| Feature | Correlation with t0 | Correlation with t1 |
|---------|---------------------|---------------------|
| p0 (best bid) | 0.133 | 0.019 |
| p2 (3rd bid) | 0.106 | 0.015 |
| v8 (3rd ask vol) | 0.097 | 0.035 |
| p10 (5th ask) | 0.082 | 0.014 |

No single feature exceeds 0.14 correlation with any target. Predictive signal exists in temporal patterns across steps and nonlinear feature interactions, not in individual snapshots.

However, feature p10 shows 0.44 correlation with lagged t0 - suggesting it partially encodes information about recent target values, making it a key input for the EMA feature pipeline.

### 3.3 Two Targets, Two Timescales

| Property | t0 | t1 |
|----------|----|----|
| Lag-1 autocorrelation | 0.749 | 0.966 |
| Useful context window | ~5 steps | ~20+ steps |
| Feature correlation | Moderate (up to 0.13) | Very weak (up to 0.035) |
| Oracle WPC | 0.68 | 0.91 |
| Prediction difficulty | Moderate | Very high |

This asymmetry drove the decision to use separate prediction heads for each target, and multi-scale temporal features (fast, slow, and ultra-slow EMAs).

---

## 4. Feature Engineering

### Design Principle

Every feature must be computable in O(1) time and O(1) memory per step. No growing buffers. Only exponential moving averages (recursive updates) and step-to-step differences.

### Feature Pipeline: 32 Raw Inputs to 158 Engineered Features

| Category | Count | Description |
|----------|-------|-------------|
| Raw features | 32 | Original LOB state passed through unchanged |
| Step-to-step deltas | 32 | Rate of change: feature[t] - feature[t-1] |
| Fast EMA (alpha = 0.3) | 10 | Short-term trend, ~3-step effective lookback |
| Slow EMA (alpha = 0.01) | 10 | Medium-term trend, ~100-step effective lookback |
| Glacial EMA (alpha = 0.003) | 10 | Long-term trend, ~333-step effective lookback |
| Slow EMA deviation | 10 | Feature - Slow EMA (mean-reversion signal) |
| Glacial EMA deviation | 10 | Feature - Glacial EMA (long-term mean-reversion) |
| Momentum | 10 | EMA of deltas (alpha = 0.05), smoothed rate-of-change |
| Volume imbalance | 6 | (bid_vol - ask_vol) / (|bid_vol| + |ask_vol|) per level |
| Bid-ask spread | 6 | ask_price - bid_price per level |
| Order flow imbalance | 6 | Change in bid volume minus change in ask volume |
| Cumulative OFI | 6 | EMA of order flow imbalance (alpha = 0.02) |
| Microprice + pressure | 2 | Volume-weighted mid-price proxy, bid volume fraction |
| Cross-features | 8 | Key interactions (price x volume, spread x imbalance) |
| **Total** | **158** | |

EMA features were computed on the 10 most informative features identified through gradient boosting importance analysis: p10, p9, p2, p5, dp0, v8, v0, v2, dp3, p8.

**Persistent state per sequence: 78 float32 values (312 bytes)**

### Why Three EMA Timescales?

The three speeds were designed to match the different autocorrelation profiles:

- **Fast (alpha = 0.3):** Captures recent changes relevant to t0 (which decorrelates within ~5 steps)
- **Slow (alpha = 0.01):** Tracks medium-term trends relevant to both targets
- **Glacial (alpha = 0.003):** Captures the ultra-persistent dynamics of t1 (which maintains 0.71 correlation at lag 10)

---

## 5. Model Architecture

### Architecture Overview

```
                        Input: 158 features per step
                                    |
                              [Layer Norm]
                                    |
                    [Conv1D Cross-Feature Mixer] --+
                         (3 layers, GELU)          |
                                    |              | (residual)
                                    +<-------------+
                                    |
                        [2-Layer GRU, hidden=128]
                                    |
                         Hidden State (256 values)
                                    |
                    +---------------+---------------+
                    |                               |
            [t0 Prediction Head]            [t1 Prediction Head]
             128 -> 64 -> 1                  128 -> 64 -> 1
             (GELU, Dropout)                 (GELU, Dropout)
                    |                               |
                Output: t0                     Output: t1
```

### Component Details

**Conv1D Cross-Feature Mixer:** Three pointwise convolution layers with GELU activation, applied across the feature dimension with a residual connection. This learns LOB spatial structure - bid-ask interactions, volume imbalance patterns, and price-volume relationships - without requiring explicit hand-engineering of every possible interaction. Inspired by the CVML (Cross-Variate Mixing Layer) approach, which demonstrated significant gains on LOB prediction tasks.

**GRU Backbone:** 2-layer Gated Recurrent Unit with hidden size 128. The hidden state acts as an implicit autoregressive mechanism - at each step it compresses all previous observations into 256 continuous values, naturally learning to track target-correlated quantities without ever seeing actual target values. This is the architectural answer to the autocorrelation insight.

**Dual Prediction Heads:** Separate MLP heads for t0 and t1, each mapping the shared GRU output through a 64-unit hidden layer. This allows each target to specialize its decoder while sharing the temporal feature extraction backbone.

### Stateful ONNX Inference

Models were exported with explicit hidden state as both input and output:

```
ONNX Inputs:   features (1 x 158)  +  hidden_state (2 x 1 x 128)
ONNX Outputs:  prediction (1 x 2)  +  new_hidden_state (2 x 1 x 128)
```

Each inference step processes only the current time step - not re-processing history. This achieves ~0.25ms per model per step, making a 10-model ensemble feasible within the time budget.

The hidden state is initialized to zeros at the start of each sequence, and the model runs through all 1,000 steps (including warmup) to build up its internal representation before predictions are scored.

---

## 6. Training Pipeline

### Loss Function: Two-Phase Training

| Phase | Epochs | Loss | Rationale |
|-------|--------|------|-----------|
| Phase 1 | First 85% | Plain MSE (predictions clipped to [-6, 6]) | Stable gradients, reliable convergence |
| Phase 2 | Final 15% | Gradual blend of MSE + Weighted Pearson Correlation | Direct metric optimization after stable initialization |

The transition ramps the Pearson weight linearly from 0 to 0.5 over 10 epochs, while MSE weight decreases from 1.0 to 0.5.

**Key finding:** Plain MSE consistently outperformed target-weighted MSE (weighting by |target|). Despite being theoretically aligned with the metric, target-weighted loss caused overfitting to extreme values and degraded generalization.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 5e-4 with cosine annealing to 1e-6 |
| Weight decay | 5e-4 |
| Batch size | 64 sequences |
| Maximum epochs | 40 |
| Early stopping | Patience = 15 (on validation WPC) |
| Gradient clipping | 1.0 (max norm) |
| Feature dropout | 5% of channels randomly zeroed |
| Feature scaling | Random per-sample scaling in [0.9, 1.1] |
| Gaussian noise | Standard deviation = 0.01 |

### Cross-Validation Strategy

5-fold cross-validation over the combined training and validation datasets (12,165 total sequences), split by sequence ID. This was a critical decision - a single random train/validation split yielded ~0.27 WPC, while proper k-fold CV produced 0.29+ per fold with much more reliable performance estimation.

---

## 7. Ensemble Strategy

### Composition

The final submission used a 10-model ensemble:

| Models | Random Seed | CV Folds | Purpose |
|--------|-------------|----------|---------|
| 5 GRU models | Seed 42 | Folds 0-4 | Base data diversity |
| 5 GRU models | Seed 123 | Folds 0-4 | Initialization diversity |

All models share the same architecture (GRU, hidden=128, 2 layers) but differ in training data splits and random initialization.

**Aggregation:** Simple averaging of all 10 model predictions.

### Ensemble Impact

| Configuration | Average WPC |
|---------------|-------------|
| Single best model | ~0.29 |
| 5-model ensemble (1 seed, 5 folds) | 0.366 |
| 10-model ensemble (2 seeds, 5 folds) | 0.376 |

Ensemble diversity was the single largest performance lever - more impactful than any architectural or hyperparameter change.

### What Did Not Improve Performance

| Approach Tried | Outcome |
|----------------|---------|
| LSTM instead of GRU | No measurable improvement |
| Larger hidden size (256) | Same individual performance as 128 |
| Upweighting t1 in loss | Degraded overall score (t1 signal too weak) |
| Autoregressive target feeding | Catastrophic error compounding |
| Target-weighted MSE | Overfitting to extreme values |
| Increased model capacity | Diminishing returns beyond h=128 |

---

## 8. Inference Pipeline

### System Overview

```
For each data point in the stream:

  1. Detect new sequence? --> Reset all state (feature engine + 10 hidden states)

  2. Feature Engineering:   32 raw features --> 158 engineered features   [0.05 ms]

  3. Model Inference:       Run all 10 ONNX models with current features  [2.5 ms]
                            (each model updates its own hidden state)

  4. Ensemble:              Average 10 predictions                        [0.01 ms]

  5. Return prediction (or None during warmup steps 0-98)
```

### Performance

| Metric | Value |
|--------|-------|
| Time per step | ~1.6 ms (budget: 2.4 ms) |
| Total inference time | ~16 minutes for 1,500 sequences |
| Time budget remaining | ~44 minutes unused |
| Submission package size | ~11 MB |
| Memory per sequence | ~10 KB (feature state + 10 hidden states) |

---

## 9. Results and Analysis

### Final Standing

| Metric | Value |
|--------|-------|
| **Final Rank** | **53 / 4,900+** |
| **Final Score** | **0.3083** |
| 1st Place Score | 0.3537 |
| 8th Place Score (prize cutoff) | 0.3307 |
| Gap to 1st | 14.7% |
| Local Validation Score | 0.3762 (t0 = 0.479, t1 = 0.274) |

### Performance by Target

| Target | Local WPC | Oracle WPC | % of Oracle |
|--------|-----------|------------|-------------|
| t0 | 0.479 | 0.68 | 70% |
| t1 | 0.274 | 0.91 | 30% |

t1 remains the clear bottleneck. Its ultra-persistent dynamics (0.966 lag-1 autocorrelation) combined with extremely weak feature correlations make it the hardest component to model. Improving t1 prediction would likely yield the largest marginal gains.

### Validation-to-Leaderboard Gap

The gap between local validation (0.376) and final leaderboard (0.308) indicates moderate distribution shift between the validation and test datasets - a common challenge in financial ML where market regimes can change.

---

## 10. Lessons Learned

**1. Ensemble discipline beats architectural novelty.** A simple 10-model average of identical architectures with different seeds and data splits outperformed every architectural variation tested.

**2. Understand your metric deeply before designing losses.** The weighted Pearson metric rewards getting large moves right. Paradoxically, training with metric-aligned weighted loss caused overfitting - plain MSE with post-hoc Pearson fine-tuning worked better.

**3. Recurrent hidden state is a powerful implicit tracker.** When the target has high autocorrelation but you can't observe it, a well-trained RNN hidden state learns to approximate the latent signal. This worked far better than explicit target reconstruction attempts.

**4. O(1) feature engineering enables scalable inference.** Multi-scale EMAs, order flow imbalance, and cross-features - all computable with constant time and memory - provided meaningful signal uplift without compromising the latency budget.

**5. Cross-validation is non-negotiable in small-data regimes.** With ~12K sequences, a single train/valid split introduced enough variance to mislead model selection by 0.02+ WPC.

### Areas for Future Exploration

- State-space models (Mamba/S4) for better long-memory modeling of t1
- Architecture diversity in the ensemble (mixing GRU with Transformer variants)
- Attention mechanisms for adaptive temporal weighting
- Post-hoc prediction calibration to reduce validation-leaderboard gap

---

*Competition: Wunder Challenge - LOB Predictorium (December 2025 - March 2026)*
*4,900+ participants worldwide | $13,600 total prize pool*
