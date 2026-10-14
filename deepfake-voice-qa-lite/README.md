---
title: Deepfake Voice Detector QA RawNetLite
emoji: 🎙️
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 6.29.1
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
---

Internal QA build: human vs AI voice detection with a ~1 MB model.

Model: RawNetLite (Di Pierno et al., "End-to-end Audio Deepfake Detection from RAW Waveforms", IJCNN 2025 workshop;
https://github.com/adipiz99/RawNetLite, MIT), fine-tuned by our team on modern commercial TTS and diverse real speech.
Bands: 0–30 Human · 30–70 Suspicious · 70–100 Synthetic.
