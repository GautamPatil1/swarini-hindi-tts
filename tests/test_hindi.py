"""Regression checks for Hindi text handling and corpus/model isolation."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training" / "scripts"))
from settings import LANG_CODE, VOICE, N_TOKEN, TEACHER_REPO, TEACHER_REVISION, RunPaths, require_cpu_audio
from wer import norm


class HindiNormalizationTests(unittest.TestCase):
    def test_preserves_hindi_vowels_and_nasalization(self):
        text = "मुझे हिंदी बोलना पसंद है"
        self.assertEqual(norm(text + "।"), unicodedata.normalize("NFC", text))
        self.assertNotEqual(norm("कि"), norm("की"))
        self.assertNotEqual(norm("है"), norm("हैं"))

    def test_digit_scripts_and_code_mixing(self):
        self.assertEqual(norm("आज २५ और 500 रुपये।"), "आज 25 और 500 रुपये")
        self.assertEqual(norm("हिंदी, English!"), "हिंदी english")

    def test_equivalent_unicode_and_joiners(self):
        self.assertEqual(norm("फ़ोन"), norm(unicodedata.normalize("NFD", "फ़ोन")))
        self.assertEqual(norm("हिं\u200dदी"), norm("हिंदी"))


class PreparedCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "training" / "data" / "hindi_sents.json"
        cls.sentences = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.metadata = json.loads(cls.path.with_suffix(".meta.json").read_text(encoding="utf-8"))

    def test_corpus_identity_and_uniqueness(self):
        self.assertEqual(len(self.sentences), 12000)
        self.assertEqual(len({text.casefold() for text in self.sentences}), 12000)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), self.metadata["source_sha256"])
        self.assertTrue(all(any("\u0900" <= c <= "\u097f" for c in text) for text in self.sentences))

    def test_held_out_paragraphs_never_enter_training(self):
        rows = self.metadata["sentences"]
        held = rows[:200]
        train = rows[200:]
        self.assertTrue(all(row["split"] == "eval" for row in held))
        self.assertTrue(all(row["split"] == "train" for row in train))
        self.assertFalse({r["document_sha256"] for r in held} & {r["document_sha256"] for r in train})
        for text, row in zip(self.sentences, rows):
            self.assertEqual(hashlib.sha256(text.casefold().encode("utf-8")).hexdigest(), row["text_sha256"])


class ArtifactIsolationTests(unittest.TestCase):
    def test_punctuation_only_chunks_are_not_spoken_as_words(self):
        from swarini import Swarini
        model = Swarini.__new__(Swarini)
        calls = []
        def phonemize(text):
            calls.append(text)
            return "n"
        model.phonemize = phonemize
        self.assertEqual(list(model._chunks("नमस्ते?।आज मौसम अच्छा है॥")), ["n", "n"])
        self.assertEqual(calls, ["नमस्ते?", "आज मौसम अच्छा है॥"])

    def test_training_rejects_audio_before_cpu_regeneration(self):
        with self.assertRaisesRegex(ValueError, "regen_audio.py"):
            require_cpu_audio([{"audio_cpu": False}])

    def test_english_teacher_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = RunPaths(Path(directory))
            paths.teacher.mkdir()
            manifest = {"lang_code": LANG_CODE, "voice": "af_heart", "n_token": N_TOKEN,
                        "teacher_repo": TEACHER_REPO, "teacher_revision": TEACHER_REVISION}
            (paths.teacher / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "voice"):
                paths.manifest()

    def test_missing_hindi_model_does_not_fall_back_to_english(self):
        from swarini import Swarini
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Missing Hindi export"):
                Swarini(model_dir=directory)

    def test_english_export_is_rejected_before_inference(self):
        from swarini import Swarini
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "swarini_int8.onnx").touch()
            (root / "config.json").write_text(json.dumps({"lang_code": "a", "voice": "af_heart"}))
            with self.assertRaisesRegex(ValueError, "Hindi hf_alpha"):
                Swarini(model_dir=root, quantized=True)


if __name__ == "__main__":
    unittest.main()
