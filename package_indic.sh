#!/usr/bin/env bash
# Create a portable source + profile/data archive; machine-specific environments are omitted.
set -euo pipefail

SWARINI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SWARINI_ARCHIVE="${1:-$SWARINI_ROOT/dist/Swarini-Indic-training.zip}"
mkdir -p -- "$(dirname -- "$SWARINI_ARCHIVE")"

python3 - "$SWARINI_ROOT" "$SWARINI_ARCHIVE" <<'PY'
from pathlib import Path
import sys
import zipfile

root = Path(sys.argv[1]).resolve()
archive = Path(sys.argv[2]).expanduser().resolve()
prefix = "Swarini-Indic-training"
fixed = [
    ".gitignore", "LICENSE", "MODEL_CARD.md", "NOTICE", "README.md", "pyproject.toml",
    "train_indic.sh", "train_hindi.sh", "package_indic.sh", "package_model.sh",
    "notebooks/swarini_indic_modal.ipynb",
    "training/README.md", "training/pyproject.toml", "training/uv.lock",
]
required = [root / name for name in fixed]
missing = [path.relative_to(root).as_posix() for path in required if not path.is_file()]
if missing:
    raise SystemExit("Missing bundle files: " + ", ".join(missing))
files = required.copy()
files.extend(path for path in (root / "training/configs").rglob("*") if path.is_file())
files.extend((root / "swarini").rglob("*.py"))
files.extend((root / "training/scripts").rglob("*.py"))
files.extend((root / "notebooks").rglob("*.ipynb"))
files.extend(path for path in (root / "training/data").iterdir() if path.is_file())
for folder in ("paper", "tests"):
    files.extend(path for path in (root / folder).rglob("*") if path.is_file())

excluded_parts = {".git", ".codex-local", ".uv-bin", ".venv", "__pycache__", "runs", "dist"}
files = sorted({path for path in files if path.is_file()
                and not excluded_parts.intersection(path.relative_to(root).parts)
                and path.suffix.lower() not in {".onnx", ".wav"}
                and path.resolve() != archive})
missing = [path.relative_to(root).as_posix() for path in files if not path.is_file()]
if missing:
    raise SystemExit("Missing bundle files: " + ", ".join(missing))

archive.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
    for path in files:
        bundle.write(path, f"{prefix}/{path.relative_to(root).as_posix()}")

print(f"Created {archive}")
print(f"Included {len(files)} source, documentation, and data files ({archive.stat().st_size / 1e6:.1f} MB compressed).")
print("Excluded local tool settings, virtual environments, caches, checkpoints, and trained runs.")
PY

if command -v sha256sum >/dev/null 2>&1; then
    SWARINI_ARCHIVE_DIR="$(dirname -- "$SWARINI_ARCHIVE")"
    SWARINI_ARCHIVE_NAME="$(basename -- "$SWARINI_ARCHIVE")"
    (cd -- "$SWARINI_ARCHIVE_DIR" && sha256sum "$SWARINI_ARCHIVE_NAME") > "$SWARINI_ARCHIVE.sha256"
elif command -v shasum >/dev/null 2>&1; then
    SWARINI_ARCHIVE_DIR="$(dirname -- "$SWARINI_ARCHIVE")"
    SWARINI_ARCHIVE_NAME="$(basename -- "$SWARINI_ARCHIVE")"
    (cd -- "$SWARINI_ARCHIVE_DIR" && shasum -a 256 "$SWARINI_ARCHIVE_NAME") > "$SWARINI_ARCHIVE.sha256"
else
    python3 - "$SWARINI_ARCHIVE" "$SWARINI_ARCHIVE.sha256" <<'PY'
import hashlib
from pathlib import Path
import sys

source, digest = map(Path, sys.argv[1:])
value = hashlib.sha256(source.read_bytes()).hexdigest()
digest.write_text(f"{value}  {source.name}\n", encoding="ascii")
PY
fi
