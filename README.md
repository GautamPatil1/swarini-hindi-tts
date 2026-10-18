# Swarini: compact Indic TTS distillation

Swarini is a profile-driven pipeline for distilling smaller, single-voice Indic TTS models from
compatible teachers. The included Hindi student uses Kokoro-82M (`lang_code="h"`, voice `hf_alpha`).
The repository also includes a wired IndicVoice loader and starter Hindi, Punjabi, and Bengali
profiles. The completed training result in this repository is Hindi with Kokoro-82M.

See the [model card](MODEL_CARD.md) for intended use, data, provenance, and evaluation limits.
The [working paper draft](paper/swarini_hindi_working_draft.md) records the method and lists the
evaluation.

## Teacher choices and current language support

The reproducible student in this repository was trained with **Kokoro-82M, Hindi language code
`h`, voice `hf_alpha`**. The pinned IndicVoice-82M v1.0 profile is experimental scaffolding. At its
pinned Hugging Face revision, its `indicvoice-v1_0.pth` and Kokoro's `kokoro-v1_0.pth` have the same SHA-256:
`496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`. It can test a
different model packaging or text/G2P frontend after that path is implemented and validated.

The generic path reads a JSON profile for teacher backend, revision, voice, G2P, symbol map, corpus,
script filtering, token count, and student presets. The current student architecture requires a
Kokoro-compatible 24 kHz teacher with matching feature shapes. See
[`training/configs/README.md`](training/configs/README.md) for setup details.

## Model exports

The repository includes a pretrained Hindi reference student in `models/hindi_hf_alpha/`:
`swarini.onnx` (36.99 MB FP32), `swarini_int8.onnx` (9.04 MB weight-only INT8), and `config.json`.
It has about 8.08 million parameters. New training runs write their own exports to
`training/runs/<profile-run-name>/models/`. Bundle any model directory with
`bash package_model.sh MODEL_DIR [OUTPUT.zip]`.

## Train a model

On Apple Silicon or Linux with an NVIDIA GPU:

```bash
bash train_indic.sh
```

The default profile reproduces the Hindi Kokoro run. The launcher installs the selected profile's
dependencies, prepares text, generates teacher targets, trains the text model and decoder, then
exports both ONNX variants. Output goes to `training/runs/<profile-run-name>/models/`. Rerun the same
command after interruption to resume.

Select another profile with `SWARINI_PROFILE=training/configs/indicvoice_punjabi.json` before running
the launcher. Start with `--pilot`. The Punjabi and Bengali files are templates, not trained results.
Run CPU corpus preparation before spending GPU time.
`bash train_hindi.sh` remains a compatibility alias for the default recipe.

The [Modal notebook](notebooks/swarini_indic_modal.ipynb) supports profile selection and staged
teacher, CPU audio-regeneration, and student-training cells. Mount persistent storage. Choose Modal
GPU/CPU/RAM in its compute settings; the notebook does not provision hardware.

For a short pipeline check, use `bash train_indic.sh --pilot`. Pilot weights are not usable speech models.

## Use the model from Python

After training:

```python
from swarini import Swarini, SAMPLE_RATE
import soundfile as sf

tts = Swarini()  # bundled FP32 Hindi model
audio = tts("नमस्ते, आप कैसे हैं?")
sf.write("hello.wav", audio, SAMPLE_RATE)
```

The CLI is also available:

```bash
training/.venv/bin/python -m swarini "नमस्ते, आप कैसे हैं?" --model-dir models/hindi_hf_alpha
```

Pass `--int8` to the CLI or `quantized=True` to `Swarini` to use the smaller INT8 export. Set
`SWARINI_MODEL_DIR` to choose a model directory by default. The exported `config.json` selects the
language's G2P and symbol map. Install the matching G2P extra for IndicVoice models with
`pip install '.[indic-g2p]'`.

## Method overview

The student has two parts: a text model predicts phoneme durations, pitch, loudness, and features; a
decoder turns those features into audio. Training first fits each part to frozen teacher outputs,
then fine-tunes the decoder adversarially. A parameter-free phase filter is included in the exported
graph. See [training/README.md](training/README.md) for architecture and commands.

## Dataset

The included Hindi text comes from public
[AI4Bharat IndicCorpV2](https://huggingface.co/datasets/ai4bharat/IndicCorpV2). Sentence selection,
source revision, and filtering are documented in [training/data/README.md](training/data/README.md).
The teacher generates the speech targets; Whisper is optional for transcription-based evaluation.

## Attribution and prior work

Swarini is a Hindi adaptation built with substantial reference to and code adaptation from
[Paradee](https://github.com/sahilmahendrakar/paradee), by Sahil Mahendrakar. See the
[original paper](https://github.com/sahilmahendrakar/paradee/blob/main/paper/Paradee-Mahendrakar-2026.pdf).
Paradee's training pipeline, student design, decoder training, phase-filter export, and runtime were
a major foundation for this project. We retain its Apache-2.0 license and cite the original work;
Swarini is not an official Paradee release.

The frozen teachers are [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) and the
profile-compatible [IndicVoice-82M](https://huggingface.co/Bindkushal/IndicVoice-82M). Frontends use
[Misaki](https://github.com/hexgrad/misaki) or [Indic G2P](https://github.com/Bindkushal/indic-g2p),
and corpora can use AI4Bharat IndicCorpV2. The Hindi student result in this checkout used Kokoro.
[Teacher-in-the-Loop](https://huggingface.co/MithrilMan/teacher-in-the-loop) is a
methodological reference for future experiments; this repository's current recipe is offline
teacher-target distillation and should not be described as a full implementation of that method.

Reference citation:

```bibtex
@misc{mahendrakar2026paradee,
  title  = {Paradee: Distilling Kokoro-82M into an 8M-Parameter Single-Voice Text-to-Speech Model},
  author = {Mahendrakar, Sahil},
  year   = {2026},
  url    = {https://github.com/sahilmahendrakar/paradee}
}

@misc{bind2026indicvoice,
  title        = {IndicVoice: Decoder-Only Neural TTS with Native G2P for Indian Languages},
  author       = {Bind, Kushal Kant},
  year         = {2026},
  institution  = {Chandigarh University},
  howpublished = {GSoC 2026, Sugar Labs},
  url          = {https://huggingface.co/Bindkushal/IndicVoice-82M}
}

@misc{angela2026teacherintheloop,
  title  = {Teacher in the Loop: Reliable Distillation of Kokoro-82M into 8M-Parameter Single-Voice Students, in English and Italian},
  author = {Angela, Fabio},
  year   = {2026},
  url    = {https://github.com/MithrilMan/teacher-in-the-loop}
}
```

## License

Code is Apache-2.0. Dataset and third-party asset terms are documented with their sources. See
[NOTICE](NOTICE) for project provenance and attribution.
