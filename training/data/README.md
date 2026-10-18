# Hindi distillation text

`hindi_sents.json` contains 12,000 Hindi sentences selected from
[AI4Bharat IndicCorpV2 on Hugging Face](https://huggingface.co/datasets/ai4bharat/IndicCorpV2).
The source dataset's [license statement](https://huggingface.co/datasets/ai4bharat/IndicCorpV2#license)
declares CC0. The selected source revision is `2d7285e6ce14fdb3fb2449c9f89427b9f582ac3f`.

This is text for generating synthetic Kokoro training targets. It does not contain recorded
speakers or promise the quality of human TTS recordings. Public access works without a token.
Preparation reads a bounded stream from `data/hi-1.txt` and closes it when enough candidates
are collected; it does not download the full 26.7 GB source file.

The corpus has 11,800 training sentences and 200 held-out sentences, placed first to match
the training scripts. Entire source paragraphs used for held-out sentences are excluded
from training. Exact normalized sentences are deduplicated across both splits.

`hindi_sents.meta.json` records the seed, source revision, paragraph hashes, source line numbers,
sentence hashes, split assignments, filters, and phoneme counts. The content is normalized to
Unicode NFC without discarding Devanagari vowel signs or nasalization. A maximum of four
sentences is selected per source paragraph. Text must contain at least 85% Devanagari letters,
4–45 words, 25–300 characters, and terminal punctuation. The preparation script rejects common
web boilerplate, URLs, repeated characters, unknown phoneme symbols, and outputs exceeding
400 phoneme symbols. Misaki `0.9.4` with eSpeak NG Hindi validates every selected sentence.

The active profile's `prepare_corpus.py` handles IndicCorpV2 language files and uses its configured
script range and G2P. The included Hindi corpus can be rebuilt with:

```bash
uv run python scripts/prepare_corpus.py --force
```

Sentence selection is automated; manual review and domain-specific additions remain useful.

## Source citation

Sumanth Doddapaneni et al. (2023). *Towards Leaving No Indic Language Behind: Building
Monolingual Corpora, Benchmark and Models for Indic Languages*.
[ACL paper](https://aclanthology.org/2023.acl-long.693/).
