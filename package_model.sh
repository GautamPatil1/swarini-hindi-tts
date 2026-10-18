#!/usr/bin/env bash
# Bundle both model precisions, config, and an optional FP32 sample.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${1:-$ROOT/models/hindi_hf_alpha}"
MODEL_LANGUAGE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("language", "Hindi"))' "$MODEL_DIR/config.json")"
ARCHIVE="${2:-$ROOT/Swarini-${MODEL_LANGUAGE}-model.zip}"

python3 - "$MODEL_DIR" "$ARCHIVE" "$ROOT" <<'PY'
from pathlib import Path
import hashlib
import sys
import zipfile

model_dir = Path(sys.argv[1]).expanduser().resolve()
archive = Path(sys.argv[2]).expanduser().resolve()
repo_root = Path(sys.argv[3]).resolve()
required = ["swarini.onnx", "swarini_int8.onnx", "config.json"]
missing = [name for name in required if not (model_dir / name).is_file()]
if missing:
    raise SystemExit(f"Missing model files under {model_dir}: {', '.join(missing)}")
import json
config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
language = config.get("language", "Hindi")
prefix = f"Swarini-{language}-model"
model_files = required + (["sample_fp32.wav"] if (model_dir / "sample_fp32.wav").is_file() else [])
docs = ["LICENSE", "NOTICE"]
missing_docs = [name for name in docs if not (repo_root / name).is_file()]
if missing_docs:
    raise SystemExit(f"Missing distribution notices under {repo_root}: {', '.join(missing_docs)}")
archive.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
    for name in model_files:
        bundle.write(model_dir / name, f"{prefix}/{name}")
    for name in docs:
        bundle.write(repo_root / name, f"{prefix}/{name}")
    if language == "Hindi":
        bundle.write(repo_root / "MODEL_CARD.md", f"{prefix}/MODEL_CARD.md")
    else:
        phonemizer = config.get("phonemizer", {})
        model_card = f"""# Swarini {language} TTS

Single-voice 24 kHz TTS student exported by the Swarini profile-driven Indic distillation pipeline.

## Model metadata

- Language: {language} (`{config.get('iso_language', 'unknown')}`)
- Voice: `{config.get('voice', 'unknown')}`
- Teacher: `{config.get('teacher_repo', 'unknown')}` at revision `{config.get('teacher_revision', 'unknown')}`
- Phonemizer: `{phonemizer.get('backend', 'unknown')}` for `{phonemizer.get('language', 'unknown')}`
- Sample rate: {config.get('sample_rate', 'unknown')} Hz
- Token inventory: {config.get('n_token', 'unknown')}

## Intended use and limits

Research prototype. Evaluate pronunciation, naturalness, and coverage with native speakers before
deployment. Training quality depends on the selected teacher, G2P, data, and profile. This package
does not claim broad language or dialect coverage.

## License and attribution

See `LICENSE` and `NOTICE`. Teacher, G2P, and text corpus have separate licenses and citations.
"""
        bundle.writestr(f"{prefix}/MODEL_CARD.md", model_card)
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_suffix(archive.suffix + ".sha256").write_text(
    f"{digest}  {archive.name}\n", encoding="ascii"
)
print(f"Created {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
if "sample_fp32.wav" not in model_files:
    print("No sample_fp32.wav found; model and config are included.")
PY
