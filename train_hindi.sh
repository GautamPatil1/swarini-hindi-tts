#!/usr/bin/env bash
# Backward-compatible launcher; profile selects the language and teacher.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/train_indic.sh" "$@"
