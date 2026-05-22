import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from src.models import create_model, GRUModel, LSTMModel
from src.features import N_FEATURES


class GRUStatefulWrapper(nn.Module):
    """Trivial wrapper so torch.onnx.export sees a forward that takes (features, hidden)
    and returns (prediction, new_hidden) -- exposing hidden state as explicit ONNX I/O."""

    def __init__(self, model: GRUModel):
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor, hidden: torch.Tensor):
        return self.model.forward_step(features, hidden)


class LSTMStatefulWrapper(nn.Module):
    """Same idea but with separate h and c tensors."""

    def __init__(self, model: LSTMModel):
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        return self.model.forward_step(features, h, c)


def export_gru_to_onnx(model: GRUModel, output_path: str, n_features: int):
    model.eval()
    wrapper = GRUStatefulWrapper(model)
    wrapper.eval()

    dummy_features = torch.randn(1, n_features)
    dummy_hidden = model.init_hidden(batch_size=1)

    # dynamo=False because the dynamo path was flaky on Windows when I tried it
    torch.onnx.export(
        wrapper,
        (dummy_features, dummy_hidden),
        output_path,
        input_names=["features", "hidden"],
        output_names=["prediction", "new_hidden"],
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported GRU to {output_path}")


def export_lstm_to_onnx(model: LSTMModel, output_path: str, n_features: int):
    model.eval()
    wrapper = LSTMStatefulWrapper(model)
    wrapper.eval()

    dummy_features = torch.randn(1, n_features)
    h, c = model.init_hidden(batch_size=1)

    torch.onnx.export(
        wrapper,
        (dummy_features, h, c),
        output_path,
        input_names=["features", "h", "c"],
        output_names=["prediction", "new_h", "new_c"],
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported LSTM to {output_path}")


def verify_onnx(model, onnx_path: str, n_features: int, arch: str):
    """Run PyTorch and ONNX on the same input and assert the outputs match."""
    import onnxruntime as ort

    model.eval()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    features = torch.randn(1, n_features)

    if arch == "gru":
        hidden = model.init_hidden(1)
        with torch.no_grad():
            pt_pred, pt_hidden = model.forward_step(features, hidden)

        ort_outputs = sess.run(None, {
            "features": features.numpy(),
            "hidden": hidden.numpy(),
        })
        ort_pred, ort_hidden = ort_outputs

        pred_diff = np.abs(pt_pred.numpy() - ort_pred).max()
        hidden_diff = np.abs(pt_hidden.numpy() - ort_hidden).max()
        print(f"GRU diffs - pred: {pred_diff:.8f}, hidden: {hidden_diff:.8f}")
        assert pred_diff < 1e-5, f"Prediction mismatch: {pred_diff}"
        assert hidden_diff < 1e-5, f"Hidden state mismatch: {hidden_diff}"

    elif arch == "lstm":
        h, c = model.init_hidden(1)
        with torch.no_grad():
            pt_pred, pt_h, pt_c = model.forward_step(features, h, c)

        ort_outputs = sess.run(None, {
            "features": features.numpy(),
            "h": h.numpy(),
            "c": c.numpy(),
        })
        ort_pred, ort_h, ort_c = ort_outputs

        pred_diff = np.abs(pt_pred.numpy() - ort_pred).max()
        h_diff = np.abs(pt_h.numpy() - ort_h).max()
        c_diff = np.abs(pt_c.numpy() - ort_c).max()
        print(f"LSTM diffs - pred: {pred_diff:.8f}, h: {h_diff:.8f}, c: {c_diff:.8f}")
        assert pred_diff < 1e-5, f"Prediction mismatch: {pred_diff}"
        assert h_diff < 1e-5, f"Hidden state mismatch: {h_diff}"
        assert c_diff < 1e-5, f"Cell state mismatch: {c_diff}"

    print("ONNX verification: PASS")


def main():
    parser = argparse.ArgumentParser(description="Export a trained checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = checkpoint["args"]

    arch = model_args["arch"]
    model = create_model(
        arch=arch,
        input_dim=N_FEATURES,
        hidden_size=model_args["hidden_size"],
        num_layers=model_args["num_layers"],
        dropout=0.0,  # graph clarity. dropout is also off via eval() anyway.
        mixer_layers=model_args["mixer_layers"],
        head_hidden=model_args["head_hidden"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.output is None:
        args.output = os.path.splitext(args.checkpoint)[0] + ".onnx"

    if arch == "gru":
        export_gru_to_onnx(model, args.output, N_FEATURES)
    elif arch == "lstm":
        export_lstm_to_onnx(model, args.output, N_FEATURES)

    if args.verify:
        verify_onnx(model, args.output, N_FEATURES, arch)

    print(f"Done. ONNX model saved to: {args.output}")
    file_size = os.path.getsize(args.output) / 1024
    print(f"File size: {file_size:.1f} KB")


if __name__ == "__main__":
    main()
