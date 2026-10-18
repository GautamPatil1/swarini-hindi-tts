# Swarini Hindi TTS

## Summary

Swarini is an experimental, single-voice Hindi text-to-speech student. It predicts speech for Hindi
phonemes using Kokoro's `h` language path and the `hf_alpha` teacher voice.

## Model details

- Architecture: compact text predictor and waveform decoder with a parameter-free phase filter.
- Parameters: approximately 8.08 million in total (about 4.23M text model and 3.85M decoder).
- Sample rate: 24 kHz, mono.
- Phoneme inventory: Kokoro's shared 178-token vocabulary.
- This repository includes FP32 `models/hindi_hf_alpha/swarini.onnx` and weight-only INT8
  `models/hindi_hf_alpha/swarini_int8.onnx` exports.
- The FP32 export is 36,986,117 bytes; INT8 is 9,037,971 bytes. The Python runtime uses FP32 by
  default. INT8 can be smaller but may sound grainier.

## Training data and supervision

The text corpus contains 12,000 filtered sentences from AI4Bharat IndicCorpV2. The training split has
11,800 sentences and the held-out text split has 200. Kokoro-82M with language `h` and voice `hf_alpha`
produces the teacher targets. Speech is synthetic teacher output, not recorded human speech.

The prepared corpus source, revision, filters, split, and license information are documented in
`training/data/README.md` in the source repository. The current recipe requests an 8,000-step text
stage, a 50,000-step decoder stage, and two 5,000-step adversarial stages. These are recipe settings,
not evidence that the model has converged or reached a quality threshold.

## Evaluation status

Initial listening found the FP32 model intelligible and promising for its size. This is informal
feedback from one listener, not a measured MOS or a controlled listening study. No validated Hindi
WER/CER, speaker-similarity score, latency benchmark, or human preference result is claimed here.
Evaluate FP32 and INT8 on the same held-out Hindi sentences before making quality claims.

## Intended use and limitations

Use this model for Hindi TTS experiments, local inference, and research into compact speech models.
It is a single-voice prototype. It has not been evaluated for accessibility, high-stakes
communication, commercial deployment, or broad dialect coverage. It may mispronounce names,
numbers, code-mixed text, and regional vocabulary.

## Provenance and license

This Hindi adaptation substantially reuses and modifies the training and inference implementation
of [Paradee](https://github.com/sahilmahendrakar/paradee), whose paper is cited in the repository
README. The frozen teacher is [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). This is not an
official release of either project. See the included `LICENSE` and `NOTICE`, plus the licenses of
the upstream teacher and dataset.

This released student was trained using Kokoro-82M, not an IndicVoice-specific run. The configurable
pipeline now includes Kokoro and IndicVoice loader adapters plus profile-driven corpus/G2P handling.
The pinned IndicVoice v1.0 checkpoint has the same SHA-256 as Kokoro's checkpoint, and new language
profiles still require phone-coverage and native-speaker validation. The included Punjabi and Bengali
profiles are templates, not trained results. Teacher-in-the-Loop is related work; this model's recipe
uses offline teacher targets and does not claim to reproduce that paper's full method.

Reference: Sahil Mahendrakar. “Paradee: Distilling Kokoro-82M into an 8M-Parameter Single-Voice
Text-to-Speech Model.” 2026. https://github.com/sahilmahendrakar/paradee

Related work: Kushal Kant Bind. “IndicVoice: Decoder-Only Neural TTS with Native G2P for Indian
Languages.” 2026. https://huggingface.co/Bindkushal/IndicVoice-82M

Related work: Fabio Angela. “Teacher in the Loop: Reliable Distillation of Kokoro-82M into
8M-Parameter Single-Voice Students, in English and Italian.” 2026.
https://github.com/MithrilMan/teacher-in-the-loop
