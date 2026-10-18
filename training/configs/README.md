# Language and teacher profiles

`train_indic.sh` reads `SWARINI_PROFILE` and uses one profile for corpus preparation, G2P, teacher
loading, training, evaluation, and export. One run produces one language and one voice. The existing
Hindi recipe remains the default and keeps its old run directory so its checkpoints can resume.

```bash
bash train_indic.sh --pilot
SWARINI_PROFILE=training/configs/indicvoice_punjabi.json bash train_indic.sh --pilot
```

The pilot uses a tiny corpus and tiny step counts. It checks code paths; it does not produce useful
speech. It may run teacher generation, so select CPU/GPU deliberately. To check corpus and phone
coverage before renting a GPU, prepare a small separate corpus first:

```bash
uv sync --project training --python 3.12 --locked --extra indicvoice
SWARINI_PROFILE=training/configs/indicvoice_punjabi.json \
  training/.venv/bin/python training/scripts/prepare_corpus.py \
  --count 32 --held-out 4 --max-source-mb 64 --force
```

Corpus preparation rejects unsupported phones and reports rejected rows. The full launcher rebuilds
this small corpus to the profile's configured count. No training starts if teacher IDs exceed the
teacher embedding, the profile token count differs, or G2P phones do not fit the teacher vocabulary.

## Supported pipeline shape

The trainer expects a teacher with Kokoro-compatible components: a 24 kHz decoder, 80 mel bins,
512-channel text features, style vectors, duration/pitch prediction, and a character-indexed phone
vocabulary whose used IDs fit `student.n_token` (maximum 512). The included backends are Kokoro and
IndicVoice. Their model wrappers expose the required interfaces; another teacher needs a loader
adapter and matching feature shapes. This is not an arbitrary architecture converter.

Phonemizer backends are `misaki_espeak` and `indic_g2p`. The profile can map symbols before they
reach the teacher/student. These mappings must preserve useful pronunciation. Unknown symbols and
out-of-range IDs fail preflight; the pipeline never silently drops phones. The exported model config
stores its G2P backend, language, symbol map, teacher, voice, and token vocabulary. Install the
matching inference extra when using an Indic G2P model:

```bash
pip install '.[indic-g2p]'
```

## Included profiles

- `hindi_kokoro.json`: current, previously trained Hindi + Kokoro `h` + `hf_alpha` recipe. Legacy
  teacher shards and `training/runs/hi_hf_alpha` remain resumable.
- `hindi_indicvoice.json`: experimental IndicVoice Hindi frontend profile. Its G2P path and model
  adapter are wired. The pinned v1.0 IndicVoice and Kokoro files have identical SHA-256
  `496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`, so this is not an
  independent teacher checkpoint result.
- `indicvoice_punjabi.json` and `indicvoice_bengali.json`: starter profiles for languages listed by
  IndicVoice. They are templates, not validated training results. Run CPU corpus preflight, review
  phone coverage, confirm a native-speaker voice is suitable, then run a tiny pilot before full work.

IndicVoice v1.0 declares `n_token=178` while its config includes IDs above 177. The pipeline filters
unused invalid IDs from the exported runtime vocabulary and rejects any high-ID phone actually
emitted by a profile's G2P. Existing Hindi symbol mappings cover the known out-of-range single
symbols for the Hindi profile. Do not remove the checks or claim a language works until preflight
and native listening succeed.

## Add another language

1. Copy the closest JSON profile to a new file. Give it a unique `id`, `run_name`, and corpus path.
2. Set the teacher repo, pinned revision, model filename, supported language code, and voice file.
3. Set a real G2P backend and language. Add explicit symbol mappings only when pronunciation stays
   defensible. Use one or more Unicode ranges for script filtering.
4. Select the dataset revision and language files. IndicCorpV2 files use names such as `data/ta.txt`,
   `data/te.txt`, and `data/gu.txt`; check the pinned dataset tree before using a path.
5. Set `student.n_token` to the teacher embedding size, 24 kHz sample rate, and student presets.
6. Run `prepare_corpus.py` on CPU. Inspect its accepted count and rejection report. Then run
   `train_indic.sh --pilot`, listen to its outputs, and verify language/voice with native speakers.
7. Start a full run only after preflight succeeds. Use a fresh profile-specific run directory.

IndicVoice currently documents Hindi, Punjabi, and Bengali. Other Indic languages need a teacher and
G2P that cover them. A text corpus alone does not provide voice support. The pipeline provides
reusable training code; language quality and teacher compatibility still need validation per
profile.
