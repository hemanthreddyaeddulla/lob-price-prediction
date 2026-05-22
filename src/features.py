import numpy as np
from typing import Optional


# raw state layout: p0-p11 prices, v0-v11 volumes, dp0-dp3 trade prices, dv0-dv3 trade volumes
P_BID = slice(0, 6)
P_ASK = slice(6, 12)
V_BID = slice(12, 18)
V_ASK = slice(18, 24)
DP = slice(24, 28)
DV = slice(28, 32)

# subset of features used for the EMA pipeline. picked by gradient-boosting importance
# (p10 is the strongest single predictor of lagged t0)
EMA_INDICES = np.array([10, 9, 2, 5, 24, 20, 12, 14, 27, 8], dtype=np.int32)
N_EMA = len(EMA_INDICES)

ALPHA_FAST = 0.3       # ~3-step memory, for t0
ALPHA_SLOW = 0.01      # ~100-step memory
ALPHA_GLACIAL = 0.003  # ~333-step memory, for the very persistent t1
ALPHA_MOMENTUM = 0.05
ALPHA_CUM_OFI = 0.02

N_RAW = 32
N_DELTA = 32
N_FAST_EMA = N_EMA
N_SLOW_EMA = N_EMA
N_GLACIAL_EMA = N_EMA
N_EMA_DEV = N_EMA
N_GLACIAL_DEV = N_EMA
N_MOMENTUM = N_EMA
N_VOL_IMB = 6
N_SPREAD = 6
N_OFI = 6
N_CUM_OFI = 6
N_MICRO = 2
N_CROSS = 8
N_FEATURES = (N_RAW + N_DELTA + N_FAST_EMA + N_SLOW_EMA + N_GLACIAL_EMA +
              N_EMA_DEV + N_GLACIAL_DEV + N_MOMENTUM + N_VOL_IMB + N_SPREAD +
              N_OFI + N_CUM_OFI + N_MICRO + N_CROSS)  # 158


class OnlineFeatureEngine:
    """O(1) feature engine. 32 raw inputs -> 158 features, recursive-only.

    Carries ~78 float32 of state (prev_state, EMAs, cum OFI). Reset on new sequence.
    """

    def __init__(self):
        self.prev_state: Optional[np.ndarray] = None
        self.fast_ema: Optional[np.ndarray] = None
        self.slow_ema: Optional[np.ndarray] = None
        self.glacial_ema: Optional[np.ndarray] = None
        self.momentum_ema: Optional[np.ndarray] = None
        self.cum_ofi: Optional[np.ndarray] = None
        self._output_buf = np.zeros(N_FEATURES, dtype=np.float32)

    def reset(self):
        self.prev_state = None
        self.fast_ema = None
        self.slow_ema = None
        self.glacial_ema = None
        self.momentum_ema = None
        self.cum_ofi = None

    @property
    def n_features(self) -> int:
        return N_FEATURES

    def process(self, state: np.ndarray) -> np.ndarray:
        s = state.astype(np.float32)
        out = self._output_buf

        out[:N_RAW] = s

        offset = N_RAW
        if self.prev_state is not None:
            deltas = s - self.prev_state
        else:
            deltas = np.zeros(N_RAW, dtype=np.float32)
        out[offset:offset + N_DELTA] = deltas

        offset += N_DELTA
        ema_values = s[EMA_INDICES]
        if self.fast_ema is None:
            self.fast_ema = ema_values.copy()
        else:
            self.fast_ema = ALPHA_FAST * ema_values + (1 - ALPHA_FAST) * self.fast_ema
        out[offset:offset + N_FAST_EMA] = self.fast_ema

        offset += N_FAST_EMA
        if self.slow_ema is None:
            self.slow_ema = ema_values.copy()
        else:
            self.slow_ema = ALPHA_SLOW * ema_values + (1 - ALPHA_SLOW) * self.slow_ema
        out[offset:offset + N_SLOW_EMA] = self.slow_ema

        offset += N_SLOW_EMA
        if self.glacial_ema is None:
            self.glacial_ema = ema_values.copy()
        else:
            self.glacial_ema = ALPHA_GLACIAL * ema_values + (1 - ALPHA_GLACIAL) * self.glacial_ema
        out[offset:offset + N_GLACIAL_EMA] = self.glacial_ema

        # mean-reversion signal: how far above/below trend are we?
        offset += N_GLACIAL_EMA
        out[offset:offset + N_EMA_DEV] = ema_values - self.slow_ema

        offset += N_EMA_DEV
        out[offset:offset + N_GLACIAL_DEV] = ema_values - self.glacial_ema

        # smoothed deltas. picks up persistent direction without short-term noise
        offset += N_GLACIAL_DEV
        if self.prev_state is not None:
            delta_ema_values = deltas[EMA_INDICES]
        else:
            delta_ema_values = np.zeros(N_EMA, dtype=np.float32)
        if self.momentum_ema is None:
            self.momentum_ema = delta_ema_values.copy()
        else:
            self.momentum_ema = ALPHA_MOMENTUM * delta_ema_values + (1 - ALPHA_MOMENTUM) * self.momentum_ema
        out[offset:offset + N_MOMENTUM] = self.momentum_ema

        offset += N_MOMENTUM
        v_bid = s[V_BID]
        v_ask = s[V_ASK]
        out[offset:offset + N_VOL_IMB] = (v_bid - v_ask) / (np.abs(v_bid) + np.abs(v_ask) + 1e-8)

        offset += N_VOL_IMB
        p_bid = s[P_BID]
        p_ask = s[P_ASK]
        spreads = p_ask - p_bid
        out[offset:offset + N_SPREAD] = spreads

        # order flow imbalance: change in resting bid volume vs change in resting ask volume
        offset += N_SPREAD
        delta_v_bid = deltas[V_BID]
        delta_v_ask = deltas[V_ASK]
        ofi = delta_v_bid - delta_v_ask
        out[offset:offset + N_OFI] = ofi

        # persistent OFI signal (EMA of the per-step OFI)
        offset += N_OFI
        if self.cum_ofi is None:
            self.cum_ofi = ofi.copy()
        else:
            self.cum_ofi = ALPHA_CUM_OFI * ofi + (1 - ALPHA_CUM_OFI) * self.cum_ofi
        out[offset:offset + N_CUM_OFI] = self.cum_ofi

        # microprice proxy. data is pre-normalized so this isn't a real microprice,
        # but the volume-weighted blend still preserves the relative-pressure signal.
        offset += N_CUM_OFI
        v0b = v_bid[0]
        v0a = v_ask[0]
        denom = np.abs(v0b) + np.abs(v0a) + 1e-8
        microprice_proxy = (p_bid[0] * np.abs(v0a) + p_ask[0] * np.abs(v0b)) / denom
        out[offset + 0] = microprice_proxy
        out[offset + 1] = np.abs(v0b) / denom

        # handpicked crosses. the 0.1 scale factors stop these from dominating the input
        offset += N_MICRO
        vol_imb_0 = (v_bid[0] - v_ask[0]) / (np.abs(v_bid[0]) + np.abs(v_ask[0]) + 1e-8)
        out[offset + 0] = s[10] * s[12] * 0.1
        out[offset + 1] = s[9] * s[20] * 0.1
        out[offset + 2] = spreads[0] * vol_imb_0
        out[offset + 3] = np.mean(p_bid) * np.mean(v_bid) * 0.1
        out[offset + 4] = np.mean(p_ask) * np.mean(v_ask) * 0.1
        out[offset + 5] = (np.sum(v_bid) - np.sum(v_ask)) / 6.0
        out[offset + 6] = np.mean(spreads)
        out[offset + 7] = s[24] * s[28] * 0.1

        self.prev_state = s.copy()

        return out.copy()

    def process_sequence(self, states: np.ndarray) -> np.ndarray:
        """Run the engine over a full (seq_len, 32) sequence. Used in training prep."""
        seq_len = states.shape[0]
        result = np.zeros((seq_len, N_FEATURES), dtype=np.float32)
        self.reset()
        for t in range(seq_len):
            result[t] = self.process(states[t])
        return result


def batch_process_sequences(all_states: np.ndarray, seq_indices: np.ndarray) -> np.ndarray:
    """Run the engine over multiple concatenated sequences. Resets state at each seq_ix boundary."""
    engine = OnlineFeatureEngine()
    result = np.zeros((len(all_states), N_FEATURES), dtype=np.float32)
    current_seq = None

    for i in range(len(all_states)):
        if seq_indices[i] != current_seq:
            engine.reset()
            current_seq = seq_indices[i]
        result[i] = engine.process(all_states[i])

    return result
