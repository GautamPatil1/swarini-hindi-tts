"""Prepare a bounded, deduplicated corpus for the active Indic language profile."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import unicodedata
from urllib.request import Request, urlopen
from huggingface_hub import hf_hub_download

from settings import (CORPUS_COUNT, CORPUS_DATASET, CORPUS_FILES, CORPUS_REVISION, PROFILE_ID, PHONEMIZER,
                      HELD_OUT, LANG_CODE, MAX_PER_DOCUMENT, MAX_SENTENCE_CHARS,
                      MAX_SOURCE_MB, MAX_WORDS, MAX_PHONEMES, MIN_SENTENCE_CHARS, MIN_WORDS,
                      N_TOKEN, SAMPLE_TEXT, SCRIPT_FRACTION, SENTENCES, TEACHER_REPO, VOICE,
                      TEACHER_REVISION, in_language_script, write_json)
from common import _make_phonemizer, phonemize, validate_teacher_config, validate_phones

SPLIT = re.compile(r"(?<=[।॥!?…])\s*|(?<=\.)\s+|\n+")
BOILERPLATE = re.compile(r"https?://|www\.|\S+@\S+|<[^>]+>|read more|और (?:पढ़ें|पढ़ें)|यह भी (?:पढ़ें|पढ़ें)|डिस्क्लेमर", re.I)


def normalize(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    return " ".join(text.split())


def suitable(text):
    if not MIN_SENTENCE_CHARS <= len(text) <= MAX_SENTENCE_CHARS or not MIN_WORDS <= len(text.split()) <= MAX_WORDS:
        return False
    if BOILERPLATE.search(text) or re.search(r"(.)\1{4,}", text):
        return False
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters or sum(in_language_script(c) for c in letters) / len(letters) < SCRIPT_FRACTION:
        return False
    return bool(re.search(r"[।॥.!?…][\"'”’)]*$", text))


def source_lines(max_bytes, input_path=None):
    """Stream only enough pinned Hugging Face text to fill the requested corpus."""
    used = 0
    sources = (str(input_path),) if input_path else CORPUS_FILES
    for filename in sources:
        if input_path:
            stream = Path(input_path).open("rb")
        else:
            url = f"https://huggingface.co/datasets/{CORPUS_DATASET}/resolve/{CORPUS_REVISION}/{filename}?download=true"
            headers = {"User-Agent": "swarini-indic-corpus/1.0"}
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if token:
                headers["Authorization"] = f"Bearer {token}"
            stream = urlopen(Request(url, headers=headers), timeout=60)
        with stream:
            for line_number, raw in enumerate(stream, 1):
                used += len(raw)
                if used > max_bytes:
                    return
                if len(raw) > 1_000_000:
                    continue
                yield filename, line_number, raw.decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=CORPUS_COUNT, help="total sentences, including held-out examples")
    parser.add_argument("--held-out", type=int, default=HELD_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-document", type=int, default=MAX_PER_DOCUMENT)
    parser.add_argument("--max-source-mb", type=int, default=MAX_SOURCE_MB)
    parser.add_argument("--out", type=Path, default=SENTENCES)
    parser.add_argument("--input", type=Path, help="optional local UTF-8 text file, one paragraph per line")
    parser.add_argument("--force", action="store_true", help="replace an existing prepared corpus")
    args = parser.parse_args()
    if not 0 < args.held_out < args.count or args.max_per_document < 1 or args.max_source_mb < 1:
        parser.error("Require 0 < --held-out < --count and positive document and byte limits.")
    metadata_path = args.out.with_suffix(".meta.json")
    if not args.force and (args.out.exists() or metadata_path.exists()):
        parser.error(f"{args.out} or its metadata exists. Use --force to rebuild it.")

    g2p = _make_phonemizer()
    config_path = hf_hub_download(TEACHER_REPO, "config.json", revision=TEACHER_REVISION)
    with open(config_path, encoding="utf-8") as response:
        config = json.load(response)
    validate_teacher_config(config)
    vocab = config["vocab"]
    embedding_size = int(config["n_token"])
    if embedding_size != N_TOKEN:
        raise ValueError(f"Profile student.n_token={N_TOKEN}, teacher config declares {embedding_size}.")
    max_phones = min(MAX_PHONEMES, int(config.get("plbert", {}).get("max_position_embeddings", 512)) - 2)
    sample_phones, _ = phonemize(SAMPLE_TEXT, g2p=g2p)
    validate_phones(sample_phones, vocab, embedding_size, max_phones)
    hf_hub_download(TEACHER_REPO, f"voices/{VOICE}.pt", revision=TEACHER_REVISION)

    rng = random.Random(args.seed)
    seen_sentences, seen_documents, documents = set(), set(), []
    rejected = Counter()
    target = args.count + args.held_out * args.max_per_document
    accepted = 0
    lines = source_lines(args.max_source_mb * 1024 * 1024, args.input)
    try:
        for filename, line_number, raw in lines:
            document = normalize(raw)
            if not document:
                continue
            document_id = hashlib.sha256(document.encode("utf-8")).hexdigest()
            if document_id in seen_documents:
                continue
            seen_documents.add(document_id)
            candidates = SPLIT.split(document)
            rng.shuffle(candidates)
            chosen = []
            for candidate in candidates:
                text = normalize(candidate)
                if not suitable(text):
                    rejected["text_filter"] += 1
                    continue
                key = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
                if key in seen_sentences:
                    rejected["duplicate_sentence"] += 1
                    continue
                phones, _ = phonemize(text, g2p=g2p)
                if not 2 <= len(phones) <= max_phones:
                    rejected["phoneme_length"] += 1
                    continue
                unknown = set(phones) - vocab.keys()
                out_of_range = {p for p in set(phones) & vocab.keys() if not 0 <= vocab[p] < embedding_size}
                if unknown or out_of_range:
                    rejected["unknown_or_out_of_range_phone"] += 1
                    continue
                seen_sentences.add(key)
                chosen.append({"text": text, "source_file": filename, "source_line": line_number,
                               "document_sha256": document_id, "text_sha256": key,
                               "phoneme_count": len(phones)})
                if len(chosen) == args.max_per_document:
                    break
            if chosen:
                documents.append(chosen)
                accepted += len(chosen)
                if accepted // 1000 != (accepted - len(chosen)) // 1000:
                    print(f"Accepted {accepted} candidate sentences from {len(documents)} source paragraphs", flush=True)
            if accepted >= target:
                break
    finally:
        lines.close()
    if accepted < target:
        raise RuntimeError(f"Only {accepted}/{target} candidates found. Rejected rows: {dict(rejected)}. "
                           "Check corpus files, script range, G2P, symbol_map, and --max-source-mb.")

    rng.shuffle(documents)
    held, train = [], []
    for document in documents:
        if len(held) < args.held_out:
            held.extend(document[:args.held_out - len(held)])
        else:
            train.extend(document)
    train = train[:args.count - args.held_out]
    if len(train) != args.count - args.held_out:
        raise RuntimeError("Insufficient training sentences after separating held-out paragraphs.")
    rng.shuffle(held)
    rng.shuffle(train)
    records = [{**row, "split": split} for split, rows in (("eval", held), ("train", train)) for row in rows]
    sentences = [row["text"] for row in records]
    write_json(args.out, sentences)
    metadata = {
        "dataset": CORPUS_DATASET if not args.input else "local",
        "dataset_url": f"https://huggingface.co/datasets/{CORPUS_DATASET}", "revision": CORPUS_REVISION,
        "lang_code": LANG_CODE, "n_token": N_TOKEN, "seed": args.seed,
        "profile_id": PROFILE_ID, "phonemizer_backend": PHONEMIZER["backend"],
        "count": len(sentences), "held_out": len(held), "train": len(train),
        "source_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "teacher_repo": TEACHER_REPO, "teacher_revision": TEACHER_REVISION,
        "max_phonemes": max_phones,
        "filters": {"min_chars": MIN_SENTENCE_CHARS, "max_chars": MAX_SENTENCE_CHARS,
                    "min_words": MIN_WORDS, "max_words": MAX_WORDS,
                    "script_fraction": SCRIPT_FRACTION, "max_per_document": args.max_per_document,
                    "unknown_phones_allowed": 0},
        "rejected": dict(rejected),
        "sentences": [{k: v for k, v in row.items() if k != "text"} for row in records],
    }
    write_json(metadata_path, metadata)
    print(f"Saved {len(train)} train + {len(held)} held-out sentences to {args.out}", flush=True)


if __name__ == "__main__":
    main()
