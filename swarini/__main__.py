"""Run an exported Swarini model: python -m swarini TEXT --model-dir MODEL_DIR -o output.wav"""
import argparse, sys, wave
import numpy as np
from . import Swarini, SAMPLE_RATE


def main():
    ap = argparse.ArgumentParser(prog="swarini", description="Speak text with Swarini and save a WAV file.")
    ap.add_argument("text", nargs="?", help="text to speak (read from stdin when omitted)")
    ap.add_argument("-o", "--out", default="swarini.wav")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--int8", action="store_true", help="use the smaller weight-only int8 model")
    ap.add_argument("--model-dir", help="directory containing exported ONNX files and config.json")
    a = ap.parse_args()
    text = a.text if a.text is not None else sys.stdin.read()
    audio = Swarini(quantized=a.int8, model_dir=a.model_dir)(text, speed=a.speed)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(a.out, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE); w.writeframes(pcm.tobytes())
    print(f"wrote {a.out} ({len(audio) / SAMPLE_RATE:.1f} s)")


if __name__ == "__main__":
    main()
