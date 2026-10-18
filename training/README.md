# Profile-driven Indic TTS distillation

This directory trains a small, single-language, single-voice student from a compatible frozen
teacher. The shipped model and known completed run are Hindi, Kokoro-82M, language code `h`, voice
`hf_alpha`. Training code now reads language, teacher, G2P, corpus, token inventory, and student
presets from a JSON profile.

This pipeline reuses the Kokoro-family student architecture. It supports 24 kHz teachers with
Kokoro-compatible text features, style, duration/pitch predictor, decoder, and a character-indexed
phone table. It does not turn arbitrary TTS architectures into compatible teachers automatically.
The Hindi Kokoro run is the only completed end-to-end result in this checkout. Hindi IndicVoice is
wired but experimental; Punjabi and Bengali profiles are starter templates, not validated models.

## One command

Default profile reproduces the existing Hindi run:

```bash
bash train_indic.sh --device auto
```

Select another profile with an environment variable:

```bash
SWARINI_PROFILE=training/configs/indicvoice_punjabi.json bash train_indic.sh --pilot
```

`train_hindi.sh` remains a backward-compatible alias. The launcher installs pinned base dependencies
and only installs the IndicVoice/Indic G2P extra when the selected profile needs it. It uses Python
3.12 and `uv.lock`. First launch needs internet. The included public datasets and teachers do not
require an HF token; setting `HF_TOKEN` can raise Hub rate limits.

The full default schedule prepares the profile corpus, generates teacher shards, regenerates
canonical CPU waveforms, trains text and acoustic students, runs adversarial decoder fine-tuning,
exports FP32 and INT8 ONNX, and synthesizes a sample. It resumes from saved shards and checkpoints.
Model files go to `training/runs/<run_name>/models/` unless `--run-dir` overrides it.

## Profile and CPU preflight

Profile files live in [`configs/`](configs/). Before a long run, check the selected profile and phone
coverage on CPU. This catches missing data files, unsupported G2P symbols, invalid token IDs, and
teacher embedding mismatches before GPU work:

```bash
uv sync --project training --python 3.12 --locked --extra indicvoice
SWARINI_PROFILE=training/configs/indicvoice_punjabi.json \
  training/.venv/bin/python training/scripts/prepare_corpus.py \
  --count 32 --held-out 4 --max-source-mb 64 --force
```

Use `--extra indicvoice` for IndicVoice profiles. Use `--extra indic-g2p` if a different teacher
profile uses the Indic G2P backend. The launcher chooses the needed extra automatically. For a full
CPU-only profile check, run `bash train_indic.sh --device cpu --prepare-only`; it prepares the
configured corpus and exits before teacher synthesis. Inspect the accepted count and rejection
report. Fix `symbol_map`, G2P, corpus files, Unicode ranges, or source size until the corpus fills.
Then `train_indic.sh --pilot` exercises a tiny teacher-to-export run. Pilot output is for pipeline
validation only.

The profile sets:

- teacher backend (`kokoro` or `indicvoice`), Hugging Face repo, pinned revision, model file, and voice;
- teacher language code, G2P backend/language, and explicit symbol replacements;
- dataset repo/revision/files, local corpus JSON path, script Unicode range, count, and held-out size;
- teacher token count, 24 kHz sample rate, and text/decoder presets.

Each new language needs a teacher that speaks it and a G2P whose phones map to valid teacher IDs.
Unknown or out-of-range phones stop corpus preparation. Do not silence those checks. See
[`configs/README.md`](configs/README.md) for requirements and included templates.

### IndicVoice v1.0 caveat

At the pinned Hugging Face revisions, IndicVoice `indicvoice-v1_0.pth` and Kokoro
`kokoro-v1_0.pth` have the same SHA-256:
`496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`. A comparison between these
v1.0 profiles tests frontend/packaging differences, not two independent trained teachers. IndicVoice
config declares `n_token=178` but contains some higher IDs. Corpus preflight accepts only phone
symbols whose actual used IDs fit the 178-row teacher embedding. The Hindi profile maps known
out-of-range symbols; the Punjabi and Bengali templates still need phone-coverage review.

## Hardware and rough time

### Apple Silicon M3, 36 GB

```bash
caffeinate -i bash train_indic.sh --device mps
```

Use macOS 14 or later, keep the Mac awake, and start with `--pilot`. MPS runs training, but teacher
audio regeneration and ONNX inference/export use CPU. The Hindi adaptation was not benchmarked on an
M3 in this VM. Full 68,000-step training may take many hours or days; record actual `s/step` on your
own M3 before planning a long run. Memory use depends on batch sizes and profile lengths.

### NVIDIA cloud GPU

```bash
bash train_indic.sh --device cuda
```

Starting recommendation: A10 24 GB or A100 40 GB, 16 CPU cores, 32–64 GiB RAM, and at least 50 GB
persistent disk. These are planning values, not measured minimums. The launcher uses larger batches
on CUDA. Keep the run directory on persistent storage.

### Modal

Notebook: [`notebooks/swarini_indic_modal.ipynb`](../notebooks/swarini_indic_modal.ipynb). Set its
`PROFILE_RELATIVE_PATH` cell variable to choose a profile. Upload the source bundle and mount the
same persistent Modal Volume for each resource change. Notebook cells resume teacher/audio/student
stages independently. Select compute in Modal's UI; the notebook does not provision hardware.

| Stage | GPU | CPUs | RAM |
|---|---:|---:|---:|
| Teacher generation | A10 24 GB or A100 40 GB | 16 | 32–64 GiB |
| Canonical audio regeneration | None | 8–16 | 8–16 GiB |
| Student + GAN training | A100 40 GB recommended | 16 | 32–64 GiB |

The previous Hindi run emitted teacher audio around 100x real time on an A100. Its CPU regeneration
rate and decoder step time depended on Modal's CPU allocation. Use your logs for cost estimates; do
not extrapolate those rates to other hardware. The training runner reports progress and saves
checkpoints every 500 steps.

### Portable source bundle

```bash
bash package_indic.sh
```

Creates `dist/Swarini-Indic-training.zip` and a SHA-256 file. It includes source, profiles, data,
docs, notebook, and lockfile. It excludes local environments, caches, model weights, and run outputs.
Unzip on macOS or Linux with internet, then run `bash train_indic.sh`. Windows needs WSL2. Copy the
whole run directory too when moving a paused run.

## Stages and resume

One run directory contains the teacher corpus, intermediate targets, checkpoints, logs, evaluation,
and export:

```text
training/runs/<run_name>/
  teacher/manifest.json, shard_*.pt, config.json
  checkpoints/text_<preset>/last.pt
  checkpoints/dec_<preset>*/last.pt
  pipeline.log, pipeline_state.json
  models/swarini.onnx, swarini_int8.onnx, config.json
```

For Modal, run teacher synthesis on GPU, wave regeneration without a GPU, then student/GAN training
on GPU. Each stage skips completed outputs. The `--teacher-only` flag generates or resumes shards
and exits. `--regen-audio-only` reads existing shards and rewrites only canonical CPU audio. Normal
launch resumes later stages. Never change profile, corpus, or run directory mid-run.

```bash
SWARINI_PROFILE=training/configs/hindi_kokoro.json bash train_indic.sh \
  --device cuda --teacher-only --run-dir /mnt/hindi-volume/hi_hf_alpha
SWARINI_PROFILE=training/configs/hindi_kokoro.json bash train_indic.sh \
  --device cpu --regen-audio-only --run-dir /mnt/hindi-volume/hi_hf_alpha
SWARINI_PROFILE=training/configs/hindi_kokoro.json bash train_indic.sh \
  --device cuda --run-dir /mnt/hindi-volume/hi_hf_alpha
```

For a Mac/cloud handoff, copy the complete run directory to persistent storage. Checkpoint files
include the optimizer and step number; rerunning the same command resumes from the last saved step.

## Corpus and objective

The default corpus is 12,000 filtered Hindi sentences from the pinned
[AI4Bharat IndicCorpV2](https://huggingface.co/datasets/ai4bharat/IndicCorpV2) revision. Profile
files can point at other IndicCorpV2 language files or a local UTF-8 source. Preparation normalizes
Unicode, filters by selected script, splits sentences, removes duplicates, holds out source
paragraphs, and validates phone IDs. Metadata records source hashes and the split.

Teacher generation stores durations, pitch, loudness, hidden text features, phonemes, and teacher
audio. Text training distills duration/prosody/hidden representations. Decoder training minimizes
log-mel and multi-resolution STFT losses, then runs two GAN refinement stages. A parameter-free
phase filter is included in ONNX export. Architecture and objective follow the Kokoro-family student
recipe; per-language quality still needs controlled listening and ASR evaluation.

## Evaluation and runtime

Training periodically evaluates held-out duration/pitch metrics and DTW log-mel error. Optional
Whisper uses a multilingual model selected with `--whisper-model small`; English-only Whisper
checkpoints are rejected. ASR measures both model errors and recognizer errors. Use native-speaker
listening for pronunciation and naturalness.

The exported `config.json` carries language, teacher, voice, G2P, symbol map, and filtered vocabulary.
Runtime picks the G2P backend from this file and fails on unknown phones rather than deleting them.
Install Indic G2P support for those exports with `pip install '.[indic-g2p]'`.

```bash
training/.venv/bin/python -m swarini "नमस्ते, आप कैसे हैं?" \
  --model-dir training/runs/hi_hf_alpha/models -o hindi.wav
```

FP32 is default. Use `--int8` for the smaller weight-only export. INT8 may add audible artifacts.
Bundle a model with `bash package_model.sh MODEL_DIR OUTPUT.zip`.

## References

- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- [IndicVoice-82M](https://huggingface.co/Bindkushal/IndicVoice-82M)
- [Indic G2P](https://github.com/Bindkushal/indic-g2p)
- [Teacher-in-the-Loop](https://huggingface.co/MithrilMan/teacher-in-the-loop)
- [Paradee](https://github.com/sahilmahendrakar/paradee), the source of substantial pipeline and student design/code
