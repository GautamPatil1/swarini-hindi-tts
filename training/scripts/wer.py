"""Multilingual Whisper word and character error rates against adjacent UTF-8 .txt references.
usage: wer.py DIR [DIR ...] [--model base] [--pattern '*student.wav']
"""
import argparse
from pathlib import Path
import unicodedata
import warnings
from settings import ISO_LANGUAGE, LANGUAGE


def norm(text):
    """Keep Unicode letters, marks, and numbers; normalize decimal digits and punctuation."""
    text = unicodedata.normalize("NFC", text).casefold()
    characters = []
    for character in text:
        if character.isdecimal():
            characters.append(str(unicodedata.decimal(character)))
        elif unicodedata.category(character)[0] in "LMN":
            characters.append(character)
        elif character in "\u200b\u200c\u200d\ufeff":
            continue
        else:
            characters.append(" ")
    return " ".join("".join(characters).split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--model", default="base", help="multilingual Whisper model, for example base, small, or large-v3")
    parser.add_argument("--pattern", default="*student.wav", help="select one system at a time, such as '*teacher.wav'")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.model.endswith(".en"):
        parser.error(f"{LANGUAGE} requires a multilingual Whisper model, not an .en model.")
    groups = []
    for directory in args.directories:
        files = sorted(directory.glob(args.pattern))
        if not files:
            parser.error(f"No WAVs matching {args.pattern!r} in {directory}.")
        references = []
        for path in files:
            reference = path.with_suffix(".txt")
            if not reference.is_file():
                parser.error(f"Missing reference transcript: {reference}")
            text = norm(reference.read_text(encoding="utf-8"))
            if not text:
                parser.error(f"Reference is empty after Unicode normalization: {reference}")
            references.append(text)
        groups.append((directory, files, references))

    warnings.filterwarnings("ignore")
    import soundfile as sf
    import torch
    import torchaudio
    import whisper
    import jiwer
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model = whisper.load_model(args.model, device=device)
    for directory, files, references in groups:
        hypotheses = []
        for path in files:
            audio, rate = sf.read(path, dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            audio = torchaudio.functional.resample(torch.tensor(audio), rate, 16000).numpy()
            result = model.transcribe(audio, language=ISO_LANGUAGE, task="transcribe", fp16=device == "cuda")
            hypotheses.append(norm(result["text"]))
        print(f"{directory}: {LANGUAGE} WER {jiwer.wer(references, hypotheses)*100:.1f}%, "
              f"CER {jiwer.cer(references, hypotheses)*100:.1f}% over {len(references)} sentences "
              f"(Whisper {args.model}, pattern {args.pattern!r})")


if __name__ == "__main__":
    main()
