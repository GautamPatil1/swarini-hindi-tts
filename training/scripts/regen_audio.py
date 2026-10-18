"""Recompute every shard's waveform with the CPU teacher decoder from the stored intermediates,
with a fixed per-sentence seed, so the corpus is deterministic and device-canonical (CPU numerics).
Shards already carrying 'audio_cpu' are skipped; the shard is rewritten atomically."""
import sys, glob, os, time, argparse
from settings import VOICE, RunPaths, add_run_argument
import torch
sys.path.insert(0, "scripts")
from common import load_teacher
from student import alignment
ap = argparse.ArgumentParser(description=__doc__)
add_run_argument(ap)
ap.add_argument("--force", action="store_true", help="regenerate rows already marked audio_cpu")
A = ap.parse_args(); P = RunPaths(A.run_dir); shards = P.shards()
pipe, m, _ = load_teacher(device="cpu", fuse=False); ref = pipe.load_voice(VOICE)
t0 = time.time(); secs = 0
for path in shards:
    path = str(path)
    rows = torch.load(path)
    if rows and all(r.get("audio_cpu") for r in rows) and not A.force: continue
    with torch.no_grad():
        for i, r in enumerate(rows):
            torch.manual_seed(r.get("seed", i)); aln = alignment(r["pred_dur"]); Fr = aln.shape[1]
            asr = r["t_en"].float()[None] @ aln; s = ref[len(r["ids"]) - 3]
            au = m.decoder(asr, r["F0"][None, :2*Fr], r["N"][None, :2*Fr], s[:, :128]).squeeze()
            r["audio_mps"] = r["audio"]; r["audio"] = (au.clamp(-1, 1) * 32767).short(); r["audio_cpu"] = True; secs += len(au) / 24000
    torch.save(rows, path + ".tmp"); os.replace(path + ".tmp", path)
    print(f"{path} {secs/3600:.2f} h  {time.time()-t0:.0f}s  {secs/(time.time()-t0):.1f}x RT", flush=True)
print("done")
