"""Shared profile settings for Kokoro-compatible Indic TTS distillation runs."""
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_ROOT.parent
DEFAULT_PROFILE_PATH = TRAINING_ROOT / "configs" / "hindi_kokoro.json"
_profile_path = Path(os.environ.get("SWARINI_PROFILE", DEFAULT_PROFILE_PATH)).expanduser()
if not _profile_path.is_absolute():
    candidates = (Path.cwd() / _profile_path, REPO_ROOT / _profile_path, TRAINING_ROOT / _profile_path)
    _profile_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
PROFILE_PATH = _profile_path.resolve()
PROFILE = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))

if PROFILE.get("schema_version") != 1:
    raise ValueError(f"Unsupported profile schema in {PROFILE_PATH}; expected schema_version=1.")

PROFILE_ID = PROFILE["id"]
LEGACY_COMPATIBLE = bool(PROFILE.get("legacy_compatible", False))
LANGUAGE = PROFILE["language"]["name"]
ISO_LANGUAGE = PROFILE["language"]["iso"]
LANG_CODE = PROFILE["language"]["teacher_code"]
UNICODE_RANGE = tuple(PROFILE["language"].get("unicode_range", (0, 0x10FFFF)))
VOICE = PROFILE["teacher"]["voice"]
TEACHER_BACKEND = PROFILE["teacher"]["backend"]
TEACHER_REPO = PROFILE["teacher"]["repo_id"]
TEACHER_REVISION = PROFILE["teacher"]["revision"]
TEACHER_MODEL_FILE = PROFILE["teacher"]["model_file"]
PHONEMIZER = PROFILE["phonemizer"]
CORPUS = PROFILE["corpus"]
CORPUS_DATASET = CORPUS["dataset"]
CORPUS_REVISION = CORPUS["revision"]
CORPUS_FILES = tuple(CORPUS["files"])
CORPUS_COUNT = int(CORPUS.get("count", 12000))
HELD_OUT = int(CORPUS.get("held_out", 200))
SCRIPT_FRACTION = float(CORPUS.get("script_fraction", 0.85))
MAX_SENTENCE_CHARS = int(CORPUS.get("max_sentence_chars", 300))
MIN_SENTENCE_CHARS = int(CORPUS.get("min_sentence_chars", 25))
MIN_WORDS = int(CORPUS.get("min_words", 4))
MAX_WORDS = int(CORPUS.get("max_words", 45))
MAX_PER_DOCUMENT = int(CORPUS.get("max_per_document", 4))
MAX_SOURCE_MB = int(CORPUS.get("max_source_mb", 64))
MAX_PHONEMES = int(CORPUS.get("max_phonemes", 400))
N_TOKEN = int(PROFILE["student"]["n_token"])
TEXT_PRESET = PROFILE["student"].get("text_preset", "s+mlp")
DECODER_PRESET = PROFILE["student"].get("decoder_preset", "A")
SAMPLE_RATE = int(PROFILE["student"].get("sample_rate", 24000))
_sentences_path = Path(os.environ.get("SWARINI_CORPUS", REPO_ROOT / PROFILE["corpus"]["path"])).expanduser()
if not _sentences_path.is_absolute():
    _sentences_path = REPO_ROOT / _sentences_path
SENTENCES = _sentences_path.resolve()
DEFAULT_RUN = TRAINING_ROOT / "runs" / PROFILE["run_name"]
SAMPLE_TEXT = PROFILE.get("sample_text", "नमस्ते, आप कैसे हैं?")
SYMBOL_MAP = PROFILE["phonemizer"].get("symbol_map", {})

if SAMPLE_RATE != 24000:
    raise ValueError("This student decoder currently supports 24 kHz Kokoro-family teachers only.")
if N_TOKEN < 1 or N_TOKEN > 512:
    raise ValueError("student.n_token must be in 1..512 for the current text encoder.")
if not 0.0 <= SCRIPT_FRACTION <= 1.0:
    raise ValueError("corpus.script_fraction must be in 0..1.")
if not 1 <= MAX_PHONEMES <= 510:
    raise ValueError("corpus.max_phonemes must be in 1..510.")


def in_language_script(char):
    """Return whether a character belongs to one of the profile's Unicode blocks."""
    blocks = UNICODE_RANGE
    if len(blocks) == 2 and all(isinstance(value, int) for value in blocks):
        blocks = (blocks,)
    return any(start <= ord(char) <= end for start, end in blocks)


def add_run_argument(parser):
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN,
                        help="isolated directory for teacher shards, checkpoints, exports, and evaluation")


def resolve_device(device="auto"):
    import torch
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable.")
    return device


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_cpu_audio(rows):
    if any(not row.get("audio_cpu") for row in rows):
        raise ValueError("Teacher audio is not canonical CPU audio. Run scripts/regen_audio.py for this --run-dir first.")


def write_transcript(waveform_path, text):
    Path(waveform_path).with_suffix(".txt").write_text(text + "\n", encoding="utf-8")


def save_checkpoint(path, value):
    """Keep the previous checkpoint intact if a run stops while saving."""
    import torch
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


@dataclass(frozen=True)
class RunPaths:
    root: Path

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    @property
    def teacher(self):
        return self.root / "teacher"

    @property
    def checkpoints(self):
        return self.root / "checkpoints"

    @property
    def models(self):
        return self.root / "models"

    @property
    def out(self):
        return self.root / "out"

    def manifest(self):
        path = self.teacher / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run scripts/gen_teacher.py for this run directory first.")
        data = json.loads(path.read_text(encoding="utf-8"))
        expected = {"lang_code": LANG_CODE, "voice": VOICE, "n_token": N_TOKEN,
                    "teacher_repo": TEACHER_REPO, "teacher_revision": TEACHER_REVISION}
        for key, value in expected.items():
            if data.get(key) != value:
                raise ValueError(f"Teacher corpus {key} is {data.get(key)!r}; expected {value!r}. Use a fresh --run-dir.")
        # Pre-profile shards are resumable only with the legacy-compatible default profile.
        if "profile_id" not in data and not LEGACY_COMPATIBLE:
            raise ValueError("This run has legacy shards without a profile identity. Use a fresh --run-dir.")
        if "profile_id" in data and data["profile_id"] != PROFILE_ID:
            raise ValueError(f"Teacher corpus profile is {data['profile_id']!r}; expected {PROFILE_ID!r}. Use a fresh --run-dir.")
        if "teacher_backend" in data and data["teacher_backend"] != TEACHER_BACKEND:
            raise ValueError("Teacher corpus was generated by another backend. Use a fresh --run-dir.")
        return data

    def shards(self, limit=None):
        self.manifest()
        paths = sorted(self.teacher.glob("shard_*.pt"))
        if limit is not None:
            paths = paths[:limit]
        if not paths:
            raise FileNotFoundError(f"No teacher shards in {self.teacher}. Run scripts/gen_teacher.py first.")
        return paths
