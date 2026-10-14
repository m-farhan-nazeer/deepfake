"""Deepfake voice detector — QA build (fine-tuned RawNetLite, ~1 MB, PyTorch on CPU).

Decision: model on the raw waveform. Preprocessing = RawNetLite's: mono 16 kHz -> peak-normalise the whole file ->
consecutive 3 s windows (a file shorter than 3 s is zero-padded) -> P(AI) per window -> mean over windows.
Bands (cut-offs chosen on a validation split, never on test data):
    P(AI) < 0.50 -> HUMAN (score 0–30) · 0.50–0.98 -> SUSPICIOUS (30–70) · >= 0.98 -> SYNTHETIC (70–100)
The waveform panel and spectrograms (linear, inverse-mel, CQT) are for display only; they don't affect the decision.
"""
from __future__ import annotations

import time

try:   # present on ZeroGPU Spaces; absent locally
    import spaces
except ImportError:
    spaces = None

import av
import gradio as gr
import librosa
import matplotlib
import numpy as np
import torch

from RawNetLite import RawNetLite   # official model definition (github.com/adipiz99/RawNetLite, MIT)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SR = 16000
SEG = 3 * SR
MAX_SECONDS = 600
SUSPICIOUS_FROM, SYNTHETIC_FROM = 0.50, 0.98
torch.set_num_threads(2)
model = RawNetLite()
model.load_state_dict(torch.load("voice_detector.pt", map_location="cpu", weights_only=True), strict=True)
model.eval()

if spaces is not None:
    # ZeroGPU refuses to start without one @spaces.GPU function; inference runs on CPU and never uses GPU quota.
    @spaces.GPU(duration=1)
    def _zerogpu_placeholder():
        return None


def load_audio(path: str) -> np.ndarray:
    limit = MAX_SECONDS * SR + SR
    with av.open(path) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SR)
        chunks, n = [], 0
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
                n += len(chunks[-1])
            if n >= limit:
                break
        else:
            for out in resampler.resample(None):
                chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError("no audio could be decoded")
    return np.concatenate(chunks).astype(np.float32)[: MAX_SECONDS * SR]


def windows(x: np.ndarray) -> np.ndarray:
    if len(x) < SEG:
        return np.pad(x, (0, SEG - len(x)))[None]
    return np.stack([x[s:s + SEG] for s in range(0, len(x) - SEG + 1, SEG)])


def band(p: float):
    if p >= SYNTHETIC_FROM:
        return "SYNTHETIC (AI)", "#dc2626", 70 + 30 * (p - SYNTHETIC_FROM) / (1 - SYNTHETIC_FROM)
    if p >= SUSPICIOUS_FROM:
        return "SUSPICIOUS", "#d97706", 30 + 40 * (p - SUSPICIOUS_FROM) / (SYNTHETIC_FROM - SUSPICIOUS_FROM)
    return "HUMAN", "#1f9d55", 30 * p / SUSPICIOUS_FROM


def inverse_mel_fb(n_fft=512, n_bands=40):
    """Mel filterbank mirrored in frequency: narrow filters at HIGH frequencies (where vocoder traces show)."""
    return librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=n_bands)[::-1, ::-1].copy()


IMFB = inverse_mel_fb()


def analysis_figure(x: np.ndarray, pw: np.ndarray):
    """Top: the waveform the model analyses, each 3 s piece coloured by its band. Below: spectrograms (display only)."""
    fig, ax = plt.subplots(4, 1, figsize=(10, 11), gridspec_kw={"height_ratios": [1.1, 1, 1, 1]})
    # 1) waveform (whole analysed file), min/max envelope so long files plot fast
    step = max(1, len(x) // 6000)
    n = len(x) // step * step
    env = x[:n].reshape(-1, step)
    t_env = np.arange(env.shape[0]) * step / SR
    ax[0].fill_between(t_env, env.min(1), env.max(1), color="#334155", lw=0)
    for i, q in enumerate(pw):
        _, color, _ = band(float(q))
        start, end = i * 3, min(i * 3 + 3, max(len(x) / SR, 3))
        ax[0].axvspan(start, end, color=color, alpha=0.25, lw=0)
        if len(pw) <= 20:
            ax[0].text((start + end) / 2, 0.82, f"{q:.3f}", ha="center", fontsize=8, color="#111")
    done = len(pw) * 3
    if len(x) / SR > done + 0.05 and len(x) >= 3 * SR:          # leftover tail < 3 s is not scored
        ax[0].axvspan(done, len(x) / SR, color="#94a3b8", alpha=0.25, lw=0)
        ax[0].text((done + len(x) / SR) / 2, -0.9, "not scored\n(< 3 s)", ha="center", fontsize=7, color="#334155")
    ax[0].set_xlim(0, max(len(x) / SR, 3))
    ax[0].set_ylim(-1.05, 1.05)
    ax[0].set_title("Sound wave the model analyses — each 3 s piece coloured by its result\n"
                    "green = Human · orange = Suspicious · red = Synthetic · numbers = P(AI)", fontsize=9)
    # 2-4) spectrograms of the first 12 s
    x = x[: 12 * SR]
    P = np.abs(librosa.stft(x, n_fft=512, hop_length=160)) ** 2
    t = len(x) / SR
    ax[1].imshow(10 * np.log10(P + 1e-10), origin="lower", aspect="auto", extent=[0, t, 0, SR / 2], cmap="magma",
                 vmin=-90, vmax=0)
    ax[1].axhspan(5000, 8000, color="cyan", alpha=0.12)
    ax[1].set_title("Linear spectrogram (first 12 s) — shaded 5–8 kHz: in our tests AI voices kept a flat\n"
                    "energy floor here, human recordings usually fade", fontsize=9)
    ax[1].set_ylabel("Hz")
    im = np.log(IMFB @ P + 1e-10)
    ax[2].imshow(im[::-1], origin="lower", aspect="auto", extent=[0, t, 0, im.shape[0]], cmap="magma")
    ax[2].set_title("Inverse-mel spectrogram — fine detail at HIGH frequencies (top rows = highest frequencies)",
                    fontsize=9)
    ax[2].set_ylabel("band")
    C = np.abs(librosa.cqt(x, sr=SR, hop_length=256, fmin=32.7, n_bins=84)) ** 2
    ax[3].imshow(10 * np.log10(C + 1e-10), origin="lower", aspect="auto", extent=[0, t, 0, 84], cmap="magma",
                 vmin=-90)
    ax[3].set_title("CQT — pitch and harmonic structure (one row per semitone, C1 to B7)", fontsize=9)
    ax[3].set_ylabel("semitone")
    ax[3].set_xlabel("seconds")
    fig.tight_layout()
    return fig


def classify(path):
    if not path:
        return "<i>Upload an audio file.</i>", "", None, None
    t0 = time.perf_counter()
    try:
        x = load_audio(path)
    except Exception as e:
        return f"<b>Could not read this file:</b> {e}", "", None, None
    if len(x) < SR // 2 or float(np.abs(x).max()) < 1e-4:
        return "<b>Audio is too short or silent.</b>", "", None, None
    x = x / float(np.abs(x).max())                       # peak normalisation, as in training
    w = windows(x)
    with torch.inference_mode():
        pw = np.concatenate([model(torch.from_numpy(w[i:i + 16]).float().unsqueeze(1)).squeeze(1).numpy()
                             for i in range(0, len(w), 16)])
    p = float(pw.mean())
    ms = (time.perf_counter() - t0) * 1000
    verdict, color, score = band(p)
    card = f"""
<div style="background:{color};color:#fff;border-radius:14px;padding:20px 24px">
  <div style="font-size:32px;font-weight:700">{verdict} · {score:.0f}/100</div>
  <div style="margin-top:6px;font-size:15px">P(AI) = <b>{p:.3f}</b> · P(human) = <b>{1 - p:.3f}</b></div>
  <div style="margin-top:6px;font-size:14px;opacity:.9">{int((pw >= SUSPICIOUS_FROM).sum())} of {len(pw)}
  three-second parts lean AI · file length {len(x) / SR:.1f} s</div>
</div>"""
    details = (f"**Processing time:** {ms:.0f} ms  \n"
               f"**Bands:** Human < {SUSPICIOUS_FROM:.2f} ≤ Suspicious < {SYNTHETIC_FROM:.2f} ≤ Synthetic (on P(AI))")
    if len(x) < 2 * SR:
        details += "  \n⚠️ Less than 2 s of audio — result is less reliable."
    rows = [[f"{3 * i}–{3 * i + 3} s", band(float(q))[0], round(float(q), 3)] for i, q in enumerate(pw)]
    fig = analysis_figure(x, pw)
    return card, details, rows, fig


with gr.Blocks(title="Deepfake Voice Detector — QA") as demo:
    gr.Markdown(
        "# Deepfake Voice Detector — QA build\n"
        "Upload a voice recording (wav, mp3, m4a, flac, ogg …). Score: **0–30 Human · 30–70 Suspicious · "
        "70–100 Synthetic**. Model: fine-tuned RawNetLite (~1 MB, runs on CPU).")
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["upload"], type="filepath", format=None, label="Audio file")
            btn = gr.Button("Analyse", variant="primary")
        with gr.Column():
            card = gr.HTML("<i>Upload an audio file.</i>")
            details = gr.Markdown()
    table = gr.Dataframe(headers=["part", "band", "P(AI)"], label="Per 3-second part", interactive=False)
    plot = gr.Plot(label="Sound wave + spectrograms (the verdict comes from the model; pictures explain it)")
    btn.click(classify, audio, [card, details, table, plot])
    audio.change(classify, audio, [card, details, table, plot])

if __name__ == "__main__":
    demo.launch()
