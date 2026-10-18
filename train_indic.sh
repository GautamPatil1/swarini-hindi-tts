#!/usr/bin/env bash
# Bootstrap the environment and run the active Indic language profile.
set -euo pipefail

SWARINI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SWARINI_ROOT"
if command -v uv >/dev/null 2>&1; then
    SWARINI_UV="$(command -v uv)"
elif [[ -x "$SWARINI_ROOT/.uv-bin/uv" ]]; then
    SWARINI_UV="$SWARINI_ROOT/.uv-bin/uv"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    SWARINI_UV="$HOME/.local/bin/uv"
else
    echo "Installing uv locally in $SWARINI_ROOT/.uv-bin"
    SWARINI_INSTALLER="$(mktemp)"
    trap 'rm -f "$SWARINI_INSTALLER"' EXIT
    curl --fail --location --silent --show-error --retry 3 https://astral.sh/uv/install.sh -o "$SWARINI_INSTALLER"
    UV_INSTALL_DIR="$SWARINI_ROOT/.uv-bin" UV_NO_MODIFY_PATH=1 sh "$SWARINI_INSTALLER"
    rm -f "$SWARINI_INSTALLER"
    trap - EXIT
    SWARINI_UV="$SWARINI_ROOT/.uv-bin/uv"
fi

PROFILE_FILE="${SWARINI_PROFILE:-$SWARINI_ROOT/training/configs/hindi_kokoro.json}"
PROFILE_EXTRAS_TEXT="$(python3 -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); t=p["teacher"]["backend"]; g=p["phonemizer"]["backend"]; print("indicvoice" if t == "indicvoice" else "indic-g2p" if g == "indic_g2p" else "")' "$PROFILE_FILE")"
PROFILE_EXTRAS=()
if [[ -n "$PROFILE_EXTRAS_TEXT" ]]; then PROFILE_EXTRAS+=(--extra "$PROFILE_EXTRAS_TEXT"); fi
"$SWARINI_UV" sync --project "$SWARINI_ROOT/training" --python 3.12 --locked "${PROFILE_EXTRAS[@]}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1
exec "$SWARINI_ROOT/training/.venv/bin/python" "$SWARINI_ROOT/training/scripts/run_indic.py" "$@"
