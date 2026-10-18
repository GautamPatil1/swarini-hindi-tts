"""Profile-configured G2P -> token IDs -> exported ONNX TTS model."""
import importlib
import json, re
import os
from pathlib import Path
import unicodedata
import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 24000
MAX_PHONEMES = 510   # the text side has 512 positions, two of which are the pad tokens at each end


class Swarini:
    """Load once, then call with text to get a float32 waveform.

    quantized: use the smaller weight-only int8 export instead of fp32.
    model_path / config_path: local export files.
    model_dir: directory containing swarini_int8.onnx, swarini.onnx, and config.json.
    Defaults to SWARINI_MODEL_DIR or this checkout's Hindi example export directory.
    threads: CPU threads for onnxruntime.
    """

    def __init__(self, quantized=False, model_path=None, config_path=None, threads=1, model_dir=None):
        name = "swarini_int8.onnx" if quantized else "swarini.onnx"
        default_dir = Path(__file__).resolve().parents[1] / "models" / "hindi_hf_alpha"
        directory = Path(model_dir or os.environ.get("SWARINI_MODEL_DIR", default_dir))
        model_path = Path(model_path) if model_path else directory / name
        config_path = Path(config_path) if config_path else model_path.with_name("config.json")
        for path in (model_path, config_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing export {path}. Follow training/README.md, then pass model_dir.")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.sample_rate = int(config.get("sample_rate", SAMPLE_RATE))
        self.n_token = int(config["n_token"])
        self.max_phonemes = min(int(config.get("max_phonemes", MAX_PHONEMES)), MAX_PHONEMES)
        if self.sample_rate != SAMPLE_RATE or not 1 <= self.n_token <= 512:
            raise ValueError("Export sample rate or token inventory does not match this runtime.")
        self.vocab = config["vocab"]
        if not self.vocab or any(not isinstance(i, int) or not 0 <= i < self.n_token for i in self.vocab.values()):
            raise ValueError("Export vocabulary contains invalid token IDs.")
        phonemizer = config.get("phonemizer", {})
        self.symbol_map = phonemizer.get("symbol_map", {})
        backend = phonemizer.get("backend")
        language = phonemizer.get("language", config.get("iso_language", "hi"))
        # Accept metadata from older Hindi exports.
        if backend is None and phonemizer.get("class", "").startswith("misaki."):
            backend = "misaki_espeak"
        if backend == "misaki_espeak":
            from misaki.espeak import EspeakG2P
            self.g2p = EspeakG2P(language=language)
        elif backend == "indic_g2p":
            module = importlib.import_module(f"indicg2p.{language}")
            self.g2p = getattr(module, phonemizer.get("class", "G2P"))()
        else:
            raise ValueError(f"Unsupported export phonemizer backend: {backend!r}.")
        if threads < 1: raise ValueError("threads must be positive.")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
    def phonemize(self, text):
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
        text = " ".join(text.split())
        result = self.g2p(text)
        phonemes = result[0] if isinstance(result, tuple) else result
        for source in sorted(self.symbol_map, key=len, reverse=True):
            phonemes = phonemes.replace(source, self.symbol_map[source])
        return phonemes

    def generate_from_phonemes(self, phonemes, speed=1.0):
        if not np.isfinite(speed) or speed <= 0: raise ValueError("speed must be finite and positive.")
        if not phonemes: return np.zeros(0, dtype=np.float32)
        unknown = sorted(set(phonemes) - self.vocab.keys())
        if unknown:
            raise ValueError(f"Phone symbols missing from export vocabulary: {unknown[:12]}")
        if len(phonemes) > self.max_phonemes:
            raise ValueError(f"Input has {len(phonemes)} phones; maximum is {self.max_phonemes}.")
        ids = [0] + [self.vocab[c] for c in phonemes] + [0]
        feed = {"input_ids": np.array([ids], dtype=np.int64), "speed": np.array([speed], dtype=np.float32)}
        return self.session.run(None, feed)[0][0]

    def __call__(self, text, speed=1.0):
        """speed > 1 speaks faster. Long text is read one sentence at a time."""
        parts = [self.generate_from_phonemes(ps, speed) for ps in self._chunks(text)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def _chunks(self, text):
        for sentence in re.split(r"(?<=[।॥!?…])\s*|(?<=\.)\s+|\n+", text.strip()):
            if not any(character.isalnum() for character in sentence):
                continue
            ps = self.phonemize(sentence)
            while len(ps) > self.max_phonemes:   # a very long sentence: cut at the last space that fits
                cut = ps.rfind(" ", 0, self.max_phonemes)
                cut = cut if cut > 0 else self.max_phonemes
                yield ps[:cut]
                ps = ps[cut:].lstrip()
            if ps:
                yield ps
