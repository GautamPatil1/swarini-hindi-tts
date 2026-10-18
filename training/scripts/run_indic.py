"""Run the active profile's Indic distillation recipe with resumable stages.

Launch with `bash train_indic.sh` from the repository root. `--pilot` exercises the
same pipeline on a tiny corpus; it does not produce a usable speech model.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from settings import (CORPUS_COUNT, DEFAULT_RUN, HELD_OUT, LANG_CODE, LANGUAGE, N_TOKEN,
                      PROFILE_ID, LEGACY_COMPATIBLE, SAMPLE_TEXT, SENTENCES,
                      TEXT_PRESET, DECODER_PRESET, TEACHER_BACKEND, PHONEMIZER,
                      TEACHER_REPO, TEACHER_REVISION,
                      TRAINING_ROOT, VOICE, write_json)

ROOT = TRAINING_ROOT.parent
SCRIPTS = Path(__file__).resolve().parent


def detect_device(requested):
    """Release PyTorch and GPU resources after detection; only stage processes need them."""
    result = subprocess.run(
        [sys.executable, "-c", "from settings import resolve_device; import sys; print(resolve_device(sys.argv[1]))", requested],
        cwd=SCRIPTS, capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "Device detection failed.")
    return result.stdout.strip()


@contextmanager
def lock_run(directory):
    """Keep two recipe processes from writing the same checkpoints on Mac/Linux."""
    import fcntl
    with (directory / "pipeline.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another recipe is running in {directory}.") from None
        yield


def pilot_corpus(directory, sentences_path):
    """Use short examples from the prepared corpus while preserving its paragraph split."""
    metadata = json.loads(sentences_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    sentences = json.loads(sentences_path.read_text(encoding="utf-8"))
    if hashlib.sha256(sentences_path.read_bytes()).hexdigest() != metadata["source_sha256"]:
        raise ValueError("Prepared corpus no longer matches its metadata.")
    nheld = metadata["held_out"]
    rows = list(zip(sentences, metadata["sentences"]))
    held = sorted(rows[:nheld], key=lambda row: row[1]["phoneme_count"])[:4]
    train = sorted(rows[nheld:], key=lambda row: row[1]["phoneme_count"])[:28]
    if len(held) != 4 or len(train) != 28:
        raise ValueError("The prepared corpus needs at least 4 held-out and 28 training sentences.")
    selected = held + train
    path = directory / "corpus" / "pilot.json"
    content = [text for text, _ in selected]
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != content:
        raise ValueError("Pilot corpus changed. Use a fresh --run-dir.")
    write_json(path, content)
    write_json(path.with_suffix(".meta.json"), {
        **{key: value for key, value in metadata.items() if key != "sentences"},
        "count": 32, "held_out": 4, "train": 28,
        "parent_source_sha256": metadata["source_sha256"],
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sentences": [row for _, row in selected],
    })
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--run-dir", type=Path, help="default: profile-specific directory under training/runs")
    parser.add_argument("--pilot", action="store_true", help="32 sentences and short training; checks the pipeline only")
    parser.add_argument("--teacher-only", action="store_true",
                        help="generate the full teacher corpus and exit without audio regeneration or student training")
    parser.add_argument("--prepare-only", action="store_true",
                        help="prepare and validate the selected text corpus, then exit before teacher synthesis")
    parser.add_argument("--regen-audio-only", action="store_true",
                        help="regenerate audio for an already generated full teacher corpus, then exit")
    parser.add_argument("--text-steps", type=int)
    parser.add_argument("--decoder-steps", type=int)
    parser.add_argument("--gan-steps", type=int, help="steps in each of the two adversarial stages")
    parser.add_argument("--text-batch-size", type=int)
    parser.add_argument("--decoder-batch-size", type=int)
    parser.add_argument("--gan-batch-size", type=int)
    parser.add_argument("--whisper-model", help="optional multilingual Whisper WER/CER, e.g. small")
    args = parser.parse_args()
    for name in ("text_steps", "decoder_steps", "gan_steps", "text_batch_size", "decoder_batch_size", "gan_batch_size"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.regen_audio_only and args.pilot:
        parser.error("--regen-audio-only is for the full teacher corpus, not a pilot.")
    if args.teacher_only and (args.regen_audio_only or args.prepare_only or args.pilot):
        parser.error("--teacher-only cannot be combined with --regen-audio-only, --prepare-only, or --pilot.")
    if args.prepare_only and (args.regen_audio_only or args.pilot):
        parser.error("--prepare-only cannot be combined with --regen-audio-only or --pilot.")
    device = detect_device(args.device)
    directory = (args.run_dir or (DEFAULT_RUN.with_name(DEFAULT_RUN.name + "_pilot") if args.pilot else DEFAULT_RUN)).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with lock_run(directory):
        run(args, device, directory)


def run(args, device, directory):
    defaults = (20, 20, 5) if args.pilot else (8000, 50000, 5000)
    steps = tuple(value if value is not None else default for value, default in
                  zip((args.text_steps, args.decoder_steps, args.gan_steps), defaults))
    batches = (2, 1, 1) if args.pilot else ((32, 16, 8) if device == "cuda" else (16, 8, 4))
    batches = tuple(value if value is not None else default for value, default in
                    zip((args.text_batch_size, args.decoder_batch_size, args.gan_batch_size), batches))
    recipe = {
        "version": 1, "pilot": args.pilot, "lang_code": LANG_CODE, "voice": VOICE,
        "n_token": N_TOKEN, "teacher_repo": TEACHER_REPO, "teacher_revision": TEACHER_REVISION,
        "text_preset": TEXT_PRESET, "decoder_preset": DECODER_PRESET, "steps": list(steps),
        "segment_frames": 16 if args.pilot else 64,
        "parent_corpus_sha256": hashlib.sha256(SENTENCES.read_bytes()).hexdigest() if SENTENCES.exists() else None,
    }
    if not LEGACY_COMPATIBLE:
        recipe.update({"profile_id": PROFILE_ID, "teacher_backend": TEACHER_BACKEND,
                       "phonemizer_backend": PHONEMIZER["backend"]})
    state_path = directory / "pipeline_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"recipe": recipe, "completed": {}}
    # A missing corpus can be rebuilt before model work, including after an interrupted download.
    if recipe["parent_corpus_sha256"] is None:
        recipe["parent_corpus_sha256"] = state["recipe"]["parent_corpus_sha256"]
    elif state["recipe"]["parent_corpus_sha256"] is None and not state["completed"].get("generate_teacher"):
        state["recipe"]["parent_corpus_sha256"] = recipe["parent_corpus_sha256"]
    if state["recipe"] != recipe:
        raise ValueError("This run has a different recipe or corpus. Use a fresh --run-dir; keep options the same to resume.")
    print(f"{LANGUAGE} / {VOICE}; device={device}; run={directory}", flush=True)
    print(f"Batch sizes: text={batches[0]}, decoder={batches[1]}, GAN={batches[2]}", flush=True)
    if args.pilot:
        print("PILOT: the resulting weights are for pipeline checks, not useful speech.", flush=True)
    if device == "cpu" and not args.regen_audio_only and not args.prepare_only:
        print("CPU selected. Full training is slow; use MPS on Mac or CUDA on an NVIDIA GPU.", flush=True)
    write_json(state_path, state)

    def stage(name, command, outputs=()):
        outputs = tuple(Path(path) for path in outputs)
        if state["completed"].get(name) and all(path.is_file() for path in outputs):
            print(f"\n[{name}] already complete", flush=True)
            return
        if name in state["completed"]:
            # Rebuilding a missing input also invalidates every later completed stage.
            completed = list(state["completed"])
            state["completed"] = {key: state["completed"][key] for key in completed[:completed.index(name)]}
            write_json(state_path, state)
        print(f"\n[{name}] {shlex.join(command)}", flush=True)
        with (directory / "pipeline.log").open("a", encoding="utf-8") as log:
            log.write(f"\n{datetime.now(timezone.utc).isoformat()} [{name}] {shlex.join(command)}\n")
            log.flush()
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if code:
            raise RuntimeError(f"Stage {name} exited with status {code}. See {directory / 'pipeline.log'}; rerun the same command to resume.")
        if any(not path.is_file() for path in outputs):
            raise RuntimeError(f"Stage {name} did not create all expected files: {outputs}")
        state["completed"][name] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)

    def script(name, *options):
        return [sys.executable, "-u", str(SCRIPTS / name), *map(str, options)]

    corpus_meta_path = SENTENCES.with_suffix(".meta.json")
    existing_count = 0
    if SENTENCES.is_file() and corpus_meta_path.is_file():
        try:
            existing_count = int(json.loads(corpus_meta_path.read_text(encoding="utf-8")).get("count", 0))
        except (ValueError, json.JSONDecodeError):
            existing_count = 0
    needed_count = 32 if args.pilot else CORPUS_COUNT
    if not SENTENCES.is_file() or not corpus_meta_path.is_file() or existing_count < needed_count:
        prep = ["--force"]
        if args.pilot and not SENTENCES.is_file():
            prep.extend(("--count", 32, "--held-out", 4))
        stage("prepare_corpus", script("prepare_corpus.py", *prep), (SENTENCES, corpus_meta_path))
        # Persist the newly generated identity before any model work begins.
        prepared_hash = hashlib.sha256(SENTENCES.read_bytes()).hexdigest()
        if recipe["parent_corpus_sha256"] not in (None, prepared_hash):
            raise ValueError("Rebuilt corpus differs from this run. Use a fresh --run-dir.")
        state["recipe"]["parent_corpus_sha256"] = prepared_hash
        write_json(state_path, state)
    corpus_meta = json.loads(SENTENCES.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if corpus_meta.get("source_sha256") != hashlib.sha256(SENTENCES.read_bytes()).hexdigest():
        raise ValueError("Prepared corpus metadata does not match its sentence file. Rebuild it with prepare_corpus.py.")
    if corpus_meta.get("profile_id") not in (None, PROFILE_ID):
        raise ValueError("Prepared corpus belongs to another profile. Use a profile-specific corpus path.")
    if "profile_id" not in corpus_meta and not LEGACY_COMPATIBLE:
        raise ValueError("Prepared corpus has no profile identity. Rebuild it with prepare_corpus.py --force.")
    if args.prepare_only:
        print(f"Corpus preflight complete: {corpus_meta['count']} sentences, profile={PROFILE_ID}, path={SENTENCES}", flush=True)
        print("No teacher synthesis, audio regeneration, or student training was started.", flush=True)
        return
    sentences = pilot_corpus(directory, SENTENCES) if args.pilot else SENTENCES
    n = 32 if args.pilot else CORPUS_COUNT
    shard_size = 16 if args.pilot else 500
    expected_shards = [directory / "teacher" / f"shard_{index:04d}.pt" for index in range((n + shard_size - 1) // shard_size)]
    shared = ("--run-dir", directory)
    if args.teacher_only:
        stage("generate_teacher", script("gen_teacher.py", n, "--sentences", sentences,
                                         "--shard-size", shard_size, "--device", device, *shared),
              (directory / "teacher" / "manifest.json", *expected_shards))
        print(f"Teacher generation complete. Run --regen-audio-only on CPU, then the normal launcher on CUDA; run={directory}", flush=True)
        return
    if args.regen_audio_only:
        missing = [path.name for path in expected_shards if not path.is_file()]
        if not state["completed"].get("generate_teacher") or missing:
            raise RuntimeError(
                "Teacher generation is not complete in this run directory. "
                "Use the normal CUDA launcher first; --regen-audio-only never generates teacher shards."
            )
        stage("regenerate_audio", script("regen_audio.py", *shared), expected_shards)
        print(f"CPU audio regeneration complete. Run the normal CUDA launcher with --run-dir {directory} to train.", flush=True)
        return
    stage("generate_teacher", script("gen_teacher.py", n, "--sentences", sentences,
                                     "--shard-size", shard_size, "--device", device, *shared),
          (directory / "teacher" / "manifest.json", *expected_shards))
    stage("regenerate_audio", script("regen_audio.py", *shared), expected_shards)
    eval_audio = 1 if args.pilot else 8
    train_options = ("--device", device, "--eval-audio", eval_audio, *shared)
    checkpoints = directory / "checkpoints"
    text = checkpoints / f"text_{TEXT_PRESET}" / "last.pt"
    decoder = checkpoints / f"dec_{DECODER_PRESET}" / "last.pt"
    gan10 = checkpoints / f"dec_{DECODER_PRESET}_gan10" / "last.pt"
    gan3 = checkpoints / f"dec_{DECODER_PRESET}_gan3" / "last.pt"
    stage("train_text", script("train_text.py", TEXT_PRESET, "--steps", steps[0], "--bs", batches[0], *train_options), (text,))
    decoder_options = ("--seg", recipe["segment_frames"], *train_options)
    stage("train_decoder", script("train_decoder.py", DECODER_PRESET, "--steps", steps[1], "--bs", batches[1], *decoder_options), (decoder,))
    for weight, initial, output in ((10, decoder, gan10), (3, gan10, gan3)):
        stage(f"train_gan{weight}", script("train_decoder.py", DECODER_PRESET, "--gan", "--mel-weight", weight,
                                          "--init-from", initial, "--steps", steps[2], "--bs", batches[2],
                                          "--tag", f"_gan{weight}", *decoder_options), (output,))
    models = directory / "models"
    stage("export", script("export_swarini.py", "--text-preset", TEXT_PRESET, "--dec-preset", DECODER_PRESET,
                            "--check-speed", 8 if args.pilot else 1, *shared),
          (models / "swarini.onnx", models / "swarini_int8.onnx", models / "config.json"))
    stage("sample", [sys.executable, "-u", "-m", "swarini", SAMPLE_TEXT, "--model-dir", str(models),
                     "--speed", "8" if args.pilot else "1", "-o", str(models / "sample_fp32.wav")],
          (models / "sample_fp32.wav",))
    if args.whisper_model:
        eval_dir = directory / "out" / f"full_{TEXT_PRESET}_{DECODER_PRESET}_gan3"
        stage("evaluation_audio", script("eval_full.py", TEXT_PRESET, DECODER_PRESET, "--dec-tag", "_gan3", "--n", eval_audio,
                                          "--device", device, *shared),
              (eval_dir / "held_1_full_student.wav",))
        for system in ("full_student", "teacher"):
            stage(f"whisper_{system}_{args.whisper_model.replace('/', '_')}", script("wer.py", eval_dir,
                  "--pattern", f"*{system}.wav", "--model", args.whisper_model))
    print(f"\nFinished. FP32 model: {models / 'swarini.onnx'}", flush=True)
    print(f"INT8 model: {models / 'swarini_int8.onnx'}", flush=True)
    print(f"Config: {models / 'config.json'}\nSample: {models / 'sample_fp32.wav'}\nLog: {directory / 'pipeline.log'}", flush=True)
    if args.pilot:
        print("Pilot only. Run without --pilot in a separate run directory for the full training recipe.", flush=True)
    else:
        print(f"Training is complete. Listen to held-out {LANGUAGE} and review ASR errors before using the model.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Rerun the same command to continue from saved checkpoints.", file=sys.stderr)
        sys.exit(130)
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"\nError: {error}", file=sys.stderr)
        sys.exit(1)
