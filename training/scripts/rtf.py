"""Single-thread CPU realtime factor: full student (text student + student decoder) vs teacher,
on the held-out sentences. usage: rtf.py TEXT_PRESET DEC_PRESET [--text-tag T] [--dec-tag T] [--threads 1]"""
import sys, glob, time, argparse, warnings
from settings import VOICE, RunPaths, add_run_argument
import torch
warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
from student import TextStudent, alignment
from student_decoder import StudentDecoder
from common import load_teacher
ap = argparse.ArgumentParser(); ap.add_argument("text"); ap.add_argument("dec"); ap.add_argument("--text-tag", default=""); ap.add_argument("--dec-tag", default=""); ap.add_argument("--threads", type=int, default=1); ap.add_argument("--n", type=int, default=8)
add_run_argument(ap)
A = ap.parse_args(); torch.set_num_threads(A.threads); P = RunPaths(A.run_dir)
held = torch.load(P.shards()[0])[:min(A.n, P.manifest()["held_out"])]
ts = TextStudent(A.text).eval(); ts.load_state_dict(torch.load(P.checkpoints / f"text_{A.text}{A.text_tag}/last.pt", map_location="cpu")["model"])
from student_decoder import load_student_decoder
ds = load_student_decoder(A.dec, P.checkpoints / f"dec_{A.dec}{A.dec_tag}/last.pt")
pipe, teacher, _ = load_teacher(device="cpu", fuse=True); ref = pipe.load_voice(VOICE)
@torch.no_grad()
def student(r):
    ids = r["ids"][None]; L = torch.tensor([len(r["ids"])])
    pd, _, _, pasr = ts(ids, L); aln = alignment(torch.round(pd[0]).clamp(min=1).long())[None]
    _, pF0, pN, _ = ts(ids, L, aln); return ds(pasr @ aln, pF0, pN).squeeze()
@torch.no_grad()
def teach(r): return teacher(r["ps"], ref[len(r["ps"]) - 1], 1.0)
for name, fn in [("teacher (82M)", teach), (f"student {A.text}+{A.dec}", student)]:
    fn(held[0]); t = time.perf_counter(); secs = 0
    for r in held: secs += len(fn(r)) / 24000
    dt = time.perf_counter() - t
    print(f"{name:28s} {secs:5.1f}s audio in {dt:6.2f}s  ->  {secs/dt:5.1f}x realtime on {A.threads} CPU thread(s)")
