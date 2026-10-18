"""Shared teacher adapters, profile G2P, quantization helpers, and metrics."""
import importlib
import warnings
from settings import (LANG_CODE, VOICE, N_TOKEN, TEACHER_BACKEND, TEACHER_MODEL_FILE,
                      TEACHER_REPO, TEACHER_REVISION, PHONEMIZER, SYMBOL_MAP,
                      SAMPLE_TEXT, resolve_device)
import numpy as np, torch, torchaudio
warnings.filterwarnings("ignore")
from huggingface_hub import hf_hub_download
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

SENTS = [SAMPLE_TEXT]


class ProfilePipeline:
    """Expose the voice/G2P methods used by training without hard-coding a language pipeline."""
    def __init__(self, g2p, voice, pack):
        self.g2p = g2p
        self.voices = {voice: pack}

    def load_voice(self, voice):
        return self.voices[voice]


def _make_phonemizer():
    backend = PHONEMIZER["backend"]
    language = PHONEMIZER.get("language", LANG_CODE)
    if backend == "misaki_espeak":
        from misaki.espeak import EspeakG2P
        return EspeakG2P(language=language)
    if backend == "indic_g2p":
        module = importlib.import_module(f"indicg2p.{language}")
        factory = getattr(module, PHONEMIZER.get("class", "G2P"))
        return factory()
    raise ValueError(f"Unsupported phonemizer backend {backend!r}; choose misaki_espeak or indic_g2p.")


def phonemize(text, pipe=None, g2p=None):
    """Return normalized phone string and backend tokens using the selected profile."""
    if pipe is not None:
        result = pipe.g2p(text)
    else:
        result = (g2p or _make_phonemizer())(text)
    phones, tokens = result if isinstance(result, tuple) else (result, None)
    if not isinstance(phones, str):
        raise TypeError(f"Phonemizer returned {type(phones).__name__}; expected a phone string.")
    for source in sorted(SYMBOL_MAP, key=len, reverse=True):
        phones = phones.replace(source, SYMBOL_MAP[source])
    return phones, tokens


def validate_teacher_config(config):
    """Fail before synthesis if profile teacher does not fit the current student architecture."""
    expected = {"n_token": N_TOKEN, "hidden_dim": 512, "style_dim": 128, "n_mels": 80}
    mismatches = {key: (config.get(key), value) for key, value in expected.items()
                  if config.get(key) != value}
    if mismatches:
        raise ValueError(f"Teacher config is incompatible with student architecture; actual/expected: {mismatches}.")


def validate_phones(phones, vocab, embedding_size, max_phones=510):
    # Each language G2P must map its phones to valid IDs in the selected teacher inventory.
    unknown = sorted(set(phones) - vocab.keys())
    out_of_range = sorted((phone, vocab[phone]) for phone in set(phones) & vocab.keys()
                          if not 0 <= vocab[phone] < embedding_size)
    if unknown or out_of_range:
        raise ValueError(f"G2P phones do not fit teacher inventory. Unknown={unknown[:12]}, "
                         f"out_of_range={out_of_range[:12]}. Fix profile G2P/symbol_map.")
    if not phones or len(phones) > max_phones:
        raise ValueError(f"G2P produced {len(phones)} phones; valid length is 1..{max_phones}.")

def load_teacher(device="cpu", voice=VOICE, fuse=True):
    device = resolve_device(device)
    config_path = hf_hub_download(TEACHER_REPO, "config.json", revision=TEACHER_REVISION)
    import json
    with open(config_path, encoding="utf-8") as handle:
        validate_teacher_config(json.load(handle))
    weights_path = hf_hub_download(TEACHER_REPO, TEACHER_MODEL_FILE, revision=TEACHER_REVISION)
    if TEACHER_BACKEND == "kokoro":
        from kokoro import KModel
        model = KModel(repo_id=TEACHER_REPO, config=config_path, model=weights_path).to(device).eval()
    elif TEACHER_BACKEND == "indicvoice":
        from indicvoice import IndicModel
        model = IndicModel(repo_id=TEACHER_REPO, config=config_path, model=weights_path).to(device).eval()
    else:
        raise ValueError(f"Unsupported teacher backend {TEACHER_BACKEND!r}.")
    embedding_size = model.text_encoder.embedding.num_embeddings
    if embedding_size != N_TOKEN:
        raise ValueError(f"Profile student.n_token={N_TOKEN}, teacher embedding has {embedding_size} rows.")
    voice_path = hf_hub_download(TEACHER_REPO, f"voices/{voice}.pt", revision=TEACHER_REVISION)
    pack = torch.load(voice_path, map_location="cpu", weights_only=True)
    pipe = ProfilePipeline(_make_phonemizer(), voice, pack)
    items = []
    for s in SENTS:
        ps, _ = phonemize(s, pipe)
        validate_phones(ps, model.vocab, embedding_size, getattr(model, "context_length", 512) - 2)
        items.append((ps, pack[len(ps) - 1]))
    if fuse:
        for _, m in model.named_modules():
            try: remove_weight_norm(m)
            except Exception: pass
            if is_parametrized(m, "weight"):
                try: remove_parametrizations(m, "weight", leave_parametrized=True)
                except Exception: pass
    return pipe, model, items

_mel = torchaudio.transforms.MelSpectrogram(24000, n_fft=1024, hop_length=256, n_mels=80)
def logmel(x): return torch.log(_mel(x).clamp_min(1e-5))

def dtw_l1(a, b):
    """Mean L1 between log-mel frames [n_mels, T] along a DTW path (timing-insensitive)."""
    a, b = a.T.numpy(), b.T.numpy()
    C = np.abs(a[:, None, :] - b[None, :, :]).mean(-1)
    n, m = C.shape; D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, path = n, m, 0
    while i > 0 and j > 0:
        path += 1; k = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return D[n, m] / path

def rmsnorm(x): return x / x.pow(2).mean().sqrt().clamp_min(1e-8)

def compare(refs, outs):
    """Returns (raw log-mel L1, DTW log-mel L1, mean |duration delta| ms). Gain-invariant (RMS-normalised)."""
    raw, dtw, ld = [], [], []
    for r, o in zip(refs, outs):
        if torch.isnan(o).any(): return float("nan"), float("nan"), float("nan")
        r, o = rmsnorm(r), rmsnorm(o)
        k = min(len(r), len(o)); lr, lo = logmel(r), logmel(o)
        raw.append((lr[:, :k // 256 + 1] - lo[:, :k // 256 + 1]).abs().mean().item())
        dtw.append(dtw_l1(lr, lo)); ld.append(abs(len(o) - len(r)) / 24)
    return float(np.mean(raw)), float(np.mean(dtw)), float(np.mean(ld))

def fq(w, bits, group=None):
    """Symmetric fake quantization, per output channel (dim 0) or per group along the rest."""
    if w.dim() < 2: return w
    qmax = 2 ** (bits - 1) - 1
    flat = w.reshape(w.shape[0], -1)
    if group:
        pad = (-flat.shape[1]) % group
        g = torch.nn.functional.pad(flat, (0, pad)).reshape(flat.shape[0], -1, group)
        s = g.abs().amax(2, keepdim=True).clamp_min(1e-8) / qmax
        out = (torch.round(g / s).clamp(-qmax, qmax) * s).reshape(flat.shape[0], -1)[:, :flat.shape[1]]
    else:
        s = flat.abs().amax(1, keepdim=True).clamp_min(1e-8) / qmax
        out = torch.round(flat / s).clamp(-qmax, qmax) * s
    return out.reshape(w.shape)

def cpu_source_stft(decoder):
    """Make an iSTFTNet decoder device-canonical: the harmonic-source STFT's raw phase feature sits on
    the ±pi branch cut for strong harmonic bins and flips sign between CPU and MPS, changing the output
    by ~0.4 DTW log-mel. Computing just that STFT on CPU makes MPS match CPU to 1e-5."""
    g = decoder.generator; st = g.stft; orig = type(st).transform
    def transform(x):
        mag, ph = orig(st, x.cpu()); return mag.to(x.device), ph.to(x.device)
    st.transform = transform
    return decoder
