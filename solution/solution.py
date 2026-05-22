import os
import sys
import glob
import json
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# DataPoint lives outside the submission zip in the eval environment. Allow either layout.
try:
    sys.path.append(f"{CURRENT_DIR}/..")
    from utils import DataPoint
except ImportError:
    DataPoint = None

sys.path.insert(0, CURRENT_DIR)
from feature_eng import OnlineFeatureEngine, N_FEATURES

import onnxruntime as ort


class PredictionModel:
    """Submission entry point. Loads every .onnx file in this directory, runs them
    all on each step (including warmup, to keep the hidden states warm), and
    averages the predictions."""

    def __init__(self):
        self.feat_eng = OnlineFeatureEngine()
        self.current_seq_ix = None

        onnx_files = sorted(glob.glob(os.path.join(CURRENT_DIR, "*.onnx")))
        if not onnx_files:
            raise FileNotFoundError(f"No .onnx files found in {CURRENT_DIR}")

        # Force single-thread. ORT will otherwise grab all cores and oversubscribe
        # on the 1-vCPU eval box.
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.models = []
        for path in onnx_files:
            sess = ort.InferenceSession(path, sess_options, providers=["CPUExecutionProvider"])
            input_names = [inp.name for inp in sess.get_inputs()]
            output_names = [out.name for out in sess.get_outputs()]
            input_shapes = {inp.name: inp.shape for inp in sess.get_inputs()}

            is_lstm = "h" in input_names and "c" in input_names
            self.models.append({
                "session": sess,
                "input_names": input_names,
                "output_names": output_names,
                "input_shapes": input_shapes,
                "is_lstm": is_lstm,
            })
            print(f"Loaded: {os.path.basename(path)} ({'LSTM' if is_lstm else 'GRU'})")

        self.hidden_states = [None] * len(self.models)

        # Optional per-model weights (produced by oof_validate.py --optimize-weights).
        # In practice equal weights won, so this file is usually absent.
        weights_path = os.path.join(CURRENT_DIR, "weights.json")
        self.model_weights = None
        if os.path.exists(weights_path):
            with open(weights_path, "r") as f:
                wdata = json.load(f)
            weight_map = dict(zip(wdata["model_ids"], wdata["weights"]))
            weights = []
            for path in onnx_files:
                model_id = os.path.basename(path).replace(".onnx", "")
                weights.append(weight_map.get(model_id, 1.0 / len(onnx_files)))
            self.model_weights = np.array(weights, dtype=np.float32)
            self.model_weights /= self.model_weights.sum()
            print("Loaded ensemble weights from weights.json")
        else:
            print("No weights.json found, using equal weights")

        print(f"Ensemble: {len(self.models)} models loaded")

    def _init_hidden(self, model_info: dict):
        if model_info["is_lstm"]:
            h_shape = model_info["input_shapes"]["h"]
            c_shape = model_info["input_shapes"]["c"]
            h_shape = [s if isinstance(s, int) else 1 for s in h_shape]
            c_shape = [s if isinstance(s, int) else 1 for s in c_shape]
            return {
                "h": np.zeros(h_shape, dtype=np.float32),
                "c": np.zeros(c_shape, dtype=np.float32),
            }
        hidden_shape = model_info["input_shapes"]["hidden"]
        hidden_shape = [s if isinstance(s, int) else 1 for s in hidden_shape]
        return {"hidden": np.zeros(hidden_shape, dtype=np.float32)}

    def _run_model(self, model_info: dict, features: np.ndarray, hidden: dict):
        sess = model_info["session"]
        feat_input = features.reshape(1, -1).astype(np.float32)

        if model_info["is_lstm"]:
            ort_inputs = {"features": feat_input, "h": hidden["h"], "c": hidden["c"]}
            pred, new_h, new_c = sess.run(None, ort_inputs)
            return pred[0], {"h": new_h, "c": new_c}

        ort_inputs = {"features": feat_input, "hidden": hidden["hidden"]}
        pred, new_hidden = sess.run(None, ort_inputs)
        return pred[0], {"hidden": new_hidden}

    def predict(self, data_point: DataPoint) -> np.ndarray:
        # New sequence -> wipe feature state and all hidden states.
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.feat_eng.reset()
            self.hidden_states = [self._init_hidden(m) for m in self.models]

        features = self.feat_eng.process(data_point.state)

        # Important: even during the 99-step warmup we run all models so their
        # hidden states accumulate context. Just don't return anything yet.
        predictions = []
        for i, model_info in enumerate(self.models):
            pred, new_hidden = self._run_model(model_info, features, self.hidden_states[i])
            self.hidden_states[i] = new_hidden
            predictions.append(pred)

        if not data_point.need_prediction:
            return None

        predictions = np.array(predictions)
        if self.model_weights is not None:
            ensemble = np.average(predictions, axis=0, weights=self.model_weights)
        else:
            ensemble = np.mean(predictions, axis=0)
        return ensemble.astype(np.float32)


if __name__ == "__main__":
    # Quick local check against valid.parquet.
    test_file = os.path.join(CURRENT_DIR, "..", "datasets", "valid.parquet")

    if not os.path.exists(test_file):
        print(f"Validation file not found: {test_file}")
        sys.exit(0)

    from utils import ScorerStepByStep

    model = PredictionModel()
    scorer = ScorerStepByStep(test_file)

    print("\nRunning validation...")
    import time
    t_start = time.time()
    results = scorer.score(model)
    elapsed = time.time() - t_start

    n_seqs = len(scorer.dataset['seq_ix'].unique())
    print("\nResults:")
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")
    print(f"\nTiming: {elapsed:.1f}s for {n_seqs} sequences")
    print(f"  Per sequence: {elapsed/n_seqs*1000:.1f}ms")
    print(f"  Estimated for 1500 seqs: {elapsed/n_seqs*1500/60:.1f} min")
