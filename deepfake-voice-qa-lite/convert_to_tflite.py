"""Convert voice_detector.pt (RawNetLite) to a TensorFlow Lite model for offline mobile inference.

The network is rebuilt with plain TensorFlow ops and the PyTorch weights are copied in, so the .tflite file uses
only built-in TFLite ops (no Flex/Select-TF ops, no extra runtime needed on the phone):
    * BatchNorm is folded into the preceding Conv1d.
    * AdaptiveAvgPool1d(64) on 48000 samples = mean over 64 non-overlapping blocks of 750 samples.
    * The bidirectional GRU is unrolled. The model only reads output[:, -1, :], so the backward direction is a
      single GRU step on the last frame (it starts from h = 0 at t = 63); the forward direction runs all 64 steps.
After conversion the TFLite model is checked against PyTorch on random and sine inputs.

Model contract (same as app.py):
    input  "audio": float32 [1, 48000] — 3 s of mono 16 kHz audio. Peak-normalise the WHOLE file first
                    (x / max|x|), then cut it into consecutive 3 s windows; a file shorter than 3 s is zero-padded.
    output "p_ai" : float32 [1, 1]     — P(AI) for that window. Average over all windows, then
                    < 0.50 HUMAN · 0.50–0.98 SUSPICIOUS · >= 0.98 SYNTHETIC.

Usage (TensorFlow needs Python 3.9–3.12):
    pip install torch tensorflow numpy ai-edge-litert
    python convert_to_tflite.py                          # -> voice_detector.tflite (float32)
    python convert_to_tflite.py --quantize fp16          # half-size weights, float compute
    python convert_to_tflite.py --quantize dynamic       # int8 weights, smallest file
"""
from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf
import torch

from RawNetLite import RawNetLite

SR = 16000
SEG = 3 * SR
POOL_OUT = 64           # AdaptiveAvgPool1d(64)


def load_torch_model(path: str) -> RawNetLite:
    model = RawNetLite()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def fold_conv_bn(conv: torch.nn.Conv1d, bn: torch.nn.BatchNorm1d):
    """Conv1d + eval-mode BatchNorm1d -> one conv. Returns TF layout kernel [k, in, out] and bias [out]."""
    w = conv.weight.detach().double().numpy()                       # [out, in, k]
    b = conv.bias.detach().double().numpy()
    scale = bn.weight.detach().double().numpy() / np.sqrt(bn.running_var.detach().double().numpy() + bn.eps)
    w = w * scale[:, None, None]
    b = (b - bn.running_mean.detach().double().numpy()) * scale + bn.bias.detach().double().numpy()
    return tf.constant(w.transpose(2, 1, 0), tf.float32), tf.constant(b, tf.float32)


def gru_weights(gru: torch.nn.GRU, suffix: str):
    """PyTorch GRU weights for one direction, transposed for x @ W. Gate order r, z, n."""
    g = lambda name: getattr(gru, name + suffix).detach().numpy()
    return (tf.constant(g("weight_ih_l0").T), tf.constant(g("bias_ih_l0")),
            tf.constant(g("weight_hh_l0").T), tf.constant(g("bias_hh_l0")))


class RawNetLiteTF(tf.Module):
    def __init__(self, m: RawNetLite, batch: int):
        super().__init__()
        self.batch = batch
        self.hidden = m.gru.hidden_size
        self.pre = fold_conv_bn(m.conv_pre, m.bn_pre)
        self.blocks = [(fold_conv_bn(rb.conv1, rb.bn1), fold_conv_bn(rb.conv2, rb.bn2))
                       for rb in (m.resblock1, m.resblock2, m.resblock3)]
        self.gru_fwd = gru_weights(m.gru, "")
        self.gru_bwd = gru_weights(m.gru, "_reverse")
        self.fc1 = (tf.constant(m.fc1.weight.detach().numpy().T), tf.constant(m.fc1.bias.detach().numpy()))
        self.fc2 = (tf.constant(m.fc2.weight.detach().numpy().T), tf.constant(m.fc2.bias.detach().numpy()))
        self.__call__ = tf.function(self._forward,
                                    input_signature=[tf.TensorSpec([batch, SEG], tf.float32, name="audio")])

    @staticmethod
    def _conv(x, kb):
        return tf.nn.conv1d(x, kb[0], stride=1, padding="SAME") + kb[1]

    def _gru_step(self, gi, h, w):
        """One PyTorch GRU step; gi = x @ W_ih + b_ih already computed."""
        _, _, w_hh, b_hh = w
        gh = tf.matmul(h, w_hh) + b_hh
        i_r, i_z, i_n = tf.split(gi, 3, axis=-1)
        h_r, h_z, h_n = tf.split(gh, 3, axis=-1)
        r = tf.sigmoid(i_r + h_r)
        z = tf.sigmoid(i_z + h_z)
        n = tf.tanh(i_n + r * h_n)
        return (1.0 - z) * n + z * h

    def _forward(self, audio):
        x = tf.expand_dims(audio, -1)                                       # [B, T, 1]  (channels-last)
        x = tf.nn.relu(self._conv(x, self.pre))                             # [B, T, 64]
        for c1, c2 in self.blocks:
            y = tf.nn.relu(self._conv(x, c1))
            x = tf.nn.relu(self._conv(y, c2) + x)
        x = tf.reshape(x, [self.batch, POOL_OUT, SEG // POOL_OUT, 64])
        seq = tf.reduce_mean(x, axis=2)                                     # [B, 64 steps, 64 feat]

        w_ih, b_ih = self.gru_fwd[0], self.gru_fwd[1]
        gi = tf.matmul(seq, w_ih) + b_ih                                    # [B, 64, 3H], all steps at once
        h = tf.zeros([self.batch, self.hidden])
        for t in range(POOL_OUT):                                           # unrolled at trace time
            h = self._gru_step(gi[:, t, :], h, self.gru_fwd)
        last = seq[:, POOL_OUT - 1, :]
        h_bwd = self._gru_step(tf.matmul(last, self.gru_bwd[0]) + self.gru_bwd[1],
                               tf.zeros([self.batch, self.hidden]), self.gru_bwd)

        x = tf.concat([h, h_bwd], axis=-1)                                  # [B, 256]
        x = tf.matmul(x, self.fc1[0]) + self.fc1[1]
        x = tf.matmul(x, self.fc2[0]) + self.fc2[1]
        return {"p_ai": tf.sigmoid(x)}                                      # [B, 1]


def convert(tf_model: RawNetLiteTF, quantize: str) -> bytes:
    fn = tf_model.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([fn], tf_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if quantize == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def verify(torch_model: RawNetLite, tflite_bytes: bytes, batch: int, n: int = 6) -> float:
    try:                                    # LiteRT is the same runtime the mobile app uses
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        Interpreter = tf.lite.Interpreter
    interp = Interpreter(model_content=tflite_bytes)
    runner = interp.get_signature_runner()
    rng = np.random.default_rng(0)
    t = np.arange(SEG, dtype=np.float32) / SR
    worst = 0.0
    for i in range(n):
        if i % 2:
            x = np.stack([np.sin(2 * np.pi * rng.uniform(80, 4000) * t) for _ in range(batch)]).astype(np.float32)
        else:
            x = rng.uniform(-1, 1, (batch, SEG)).astype(np.float32)
        x /= np.abs(x).max(axis=1, keepdims=True)                          # peak-normalised, like app.py
        with torch.inference_mode():
            ref = torch_model(torch.from_numpy(x).unsqueeze(1)).numpy()
        out = runner(audio=x)["p_ai"]
        diff = float(np.abs(out - ref).max())
        worst = max(worst, diff)
        print(f"  check {i + 1}: torch={ref.ravel()[0]:.6f}  tflite={out.ravel()[0]:.6f}  |diff|={diff:.2e}")
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="voice_detector.pt")
    ap.add_argument("--out", default="voice_detector.tflite")
    ap.add_argument("--quantize", choices=["none", "fp16", "dynamic"], default="none")
    ap.add_argument("--batch", type=int, default=1, help="fixed batch size baked into the model (default 1)")
    args = ap.parse_args()

    torch_model = load_torch_model(args.weights)
    tf_model = RawNetLiteTF(torch_model, args.batch)
    tflite_bytes = convert(tf_model, args.quantize)
    with open(args.out, "wb") as f:
        f.write(tflite_bytes)
    print(f"Wrote {args.out} ({len(tflite_bytes) / 1024:.0f} KB, quantize={args.quantize}, "
          f"input [{args.batch}, {SEG}] float32 -> output [{args.batch}, 1] P(AI))")

    worst = verify(torch_model, tflite_bytes, args.batch)
    tol = 1e-4 if args.quantize == "none" else 2e-2
    print(f"Max |TFLite - PyTorch| = {worst:.2e} -> {'OK' if worst <= tol else 'MISMATCH (check before shipping)'}")


if __name__ == "__main__":
    main()
