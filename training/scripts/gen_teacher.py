"""Run the active profile's teacher over its text corpus and save resumable training shards."""
import sys, os, json, time, warnings, argparse, hashlib
from pathlib import Path
from settings import (CORPUS_COUNT, LANG_CODE, VOICE, N_TOKEN, MAX_PHONEMES, PROFILE_ID, TEACHER_BACKEND, PHONEMIZER,
                      TEACHER_REPO, TEACHER_REVISION, SENTENCES, LEGACY_COMPATIBLE,
                      RunPaths, add_run_argument, resolve_device, write_json)
import torch
warnings.filterwarnings("ignore")
from common import load_teacher, phonemize
from huggingface_hub import hf_hub_download

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("n_sentences", nargs="?", type=int, default=CORPUS_COUNT)
ap.add_argument("--sentences", type=Path, default=SENTENCES)
ap.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
ap.add_argument("--shard-size", type=int, default=500)
ap.add_argument("--held-out", type=int, help="defaults to prepared corpus metadata, otherwise 200")
add_run_argument(ap)
A = ap.parse_args(); N = A.n_sentences; SHARD = A.shard_size; P = RunPaths(A.run_dir)
if not A.sentences.exists():
    ap.error(f"Missing {A.sentences}. Run scripts/prepare_corpus.py first.")
source_bytes = A.sentences.read_bytes(); source_hash = hashlib.sha256(source_bytes).hexdigest()
sents = json.loads(source_bytes)
metadata_path = A.sentences.with_suffix(".meta.json")
metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
if metadata and metadata.get("source_sha256") != source_hash:
    ap.error("Sentence file no longer matches its metadata. Rebuild it with scripts/prepare_corpus.py.")
if metadata.get("profile_id") not in (None, PROFILE_ID):
    ap.error("Prepared corpus belongs to another profile. Use its corpus path or rebuild it with --force.")
if metadata and "profile_id" not in metadata and not LEGACY_COMPATIBLE:
    ap.error("Prepared corpus has no profile identity. Rebuild it with scripts/prepare_corpus.py --force.")
held_out = A.held_out if A.held_out is not None else metadata.get("held_out", 200)
if metadata and held_out != metadata["held_out"]:
    ap.error("--held-out differs from the prepared split. Prepare a separate corpus for a pilot run.")
if not 0 < held_out < N <= len(sents) or SHARD <= held_out:
    ap.error("Require 0 < held-out < n_sentences <= corpus size, and shard-size > held-out.")
if not isinstance(sents, list) or not all(isinstance(s, str) and s.strip() for s in sents[:N]):
    ap.error("Sentence JSON must contain a list of nonempty strings.")
sents = sents[:N]
manifest = {"lang_code": LANG_CODE, "voice": VOICE, "n_token": N_TOKEN,
            "teacher_repo": TEACHER_REPO, "teacher_revision": TEACHER_REVISION,
            "sentences_sha256": source_hash, "n_sentences": N, "shard_size": SHARD, "held_out": held_out,
            "profile_id": PROFILE_ID, "teacher_backend": TEACHER_BACKEND,
            "phonemizer_backend": PHONEMIZER["backend"]}
manifest_path = P.teacher / "manifest.json"
if manifest_path.exists():
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(old_manifest.get(key) != value for key, value in manifest.items()
           if key not in ("profile_id", "teacher_backend", "phonemizer_backend")):
        ap.error("This run belongs to another corpus or configuration. Use a fresh --run-dir.")
    if old_manifest.get("profile_id") not in (None, PROFILE_ID):
        ap.error("This run belongs to another profile. Use a fresh --run-dir.")
    if "profile_id" not in old_manifest and not LEGACY_COMPATIBLE:
        ap.error("Legacy shards have no profile identity. Use a fresh --run-dir.")
    manifest = old_manifest
elif list(P.teacher.glob("shard_*.pt")):
    ap.error("Existing shards have no manifest. Use a fresh --run-dir.")
DEV = resolve_device(A.device)
pipe, m, _ = load_teacher(device=DEV, fuse=False)
ref = pipe.load_voice(VOICE)
config_path = hf_hub_download(TEACHER_REPO, "config.json", revision=TEACHER_REVISION)
write_json(P.teacher / "config.json", json.loads(Path(config_path).read_text(encoding="utf-8")))
write_json(manifest_path, manifest)
print(f"Teacher: {LANG_CODE}/{VOICE}, device={DEV}, output={P.teacher}", flush=True)

@torch.no_grad()
def run(ps, seed):
    ids = torch.LongTensor([[0, *[m.vocab[c] for c in ps], 0]]).to(m.device)
    T = ids.shape[1]; r = ref[T - 3].to(m.device)  # style indexed by phoneme count (== T-2, minus 1)
    L = torch.full((1,), T, device=m.device); mask = torch.zeros(1, T, dtype=torch.bool, device=m.device)
    bert_dur = m.bert(ids, attention_mask=(~mask).int()); d_en = m.bert_encoder(bert_dur).transpose(-1, -2)
    s = r[:, 128:]; d = m.predictor.text_encoder(d_en, s, L, mask); x, _ = m.predictor.lstm(d)
    dur = torch.sigmoid(m.predictor.duration_proj(x)).sum(-1)
    pred_dur = torch.round(dur).clamp(min=1).long().squeeze(0)
    idx = torch.repeat_interleave(torch.arange(T, device=m.device), pred_dur)
    aln = torch.zeros(T, idx.shape[0], device=m.device); aln[idx, torch.arange(idx.shape[0])] = 1; aln = aln[None]
    en = d.transpose(-1, -2) @ aln; F0, Nn = m.predictor.F0Ntrain(en, s)
    t_en = m.text_encoder(ids, L, mask); asr = t_en @ aln
    torch.manual_seed(seed)
    audio = m.decoder(asr, F0, Nn, r[:, :128]).squeeze()
    c = lambda t: t.squeeze(0).cpu()
    return dict(ids=c(ids), dur=c(dur), pred_dur=pred_dur.cpu(), d=c(d).half(), t_en=c(t_en).half(),
                F0=c(F0), N=c(Nn), audio=(audio.cpu().clamp(-1, 1) * 32767).short(),
                seed=seed, audio_cpu=False)

t0 = time.time(); secs = 0.0
for sh in range(0, len(sents), SHARD):
    path = P.teacher / f"shard_{sh // SHARD:04d}.pt"
    if os.path.exists(path): continue
    rows = []
    for index, text in enumerate(sents[sh:sh + SHARD], sh):
        ps, _ = phonemize(text, pipe)
        if not ps or len(ps) > MAX_PHONEMES or set(ps) - m.vocab.keys():
            raise ValueError(f"Invalid phonemes for sentence {index}; rebuild the corpus instead of changing its split.")
        r = run(ps, index); r["text"] = text; r["ps"] = ps; rows.append(r); secs += len(r["audio"]) / 24000
    temporary = path.with_suffix(".pt.tmp")
    torch.save(rows, temporary); temporary.replace(path)
    print(f"{path} {len(rows)} sents  {secs/3600:.2f} h audio  {time.time()-t0:.0f}s elapsed  {secs/(time.time()-t0):.1f}x RT", flush=True)
print("done")
