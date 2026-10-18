"""Weight-only int8 (per output channel) of the full student; reports bytes on disk (int8 matrices +
fp16 everything else) and UTMOS / DTW vs the unquantized student on held-out sentences.
usage: quantize_student.py TEXT_PRESET DEC_PRESET [--text-tag T] [--dec-tag T] [--bits 8] [--dec-bits 8]"""
import sys, glob, os, argparse, warnings, torch, soundfile as sf
from settings import RunPaths, add_run_argument, write_transcript
warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
from student import TextStudent, alignment
from student_decoder import StudentDecoder
from common import fq, compare
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations
ap = argparse.ArgumentParser(); ap.add_argument("text"); ap.add_argument("dec"); ap.add_argument("--text-tag", default=""); ap.add_argument("--dec-tag", default="")
ap.add_argument("--bits", type=int, default=8); ap.add_argument("--dec-bits", type=int, default=8); ap.add_argument("--n", type=int, default=8)
add_run_argument(ap)
A = ap.parse_args(); P = RunPaths(A.run_dir)
held = torch.load(P.shards()[0])[:min(A.n, P.manifest()["held_out"])]
ts = TextStudent(A.text).eval(); ts.load_state_dict(torch.load(P.checkpoints / f"text_{A.text}{A.text_tag}/last.pt", map_location="cpu")["model"])
from student_decoder import load_student_decoder
ds = load_student_decoder(A.dec, P.checkpoints / f"dec_{A.dec}{A.dec_tag}/last.pt")
for m in list(ts.modules()) + list(ds.modules()):
    try: remove_weight_norm(m)
    except Exception: pass
    if is_parametrized(m, "weight"):
        try: remove_parametrizations(m, "weight", leave_parametrized=True)
        except Exception: pass
@torch.no_grad()
def synth(r):
    ids = r["ids"][None]; L = torch.tensor([len(r["ids"])])
    pd, _, _, pasr = ts(ids, L); aln = alignment(torch.round(pd[0]).clamp(min=1).long())[None]
    _, pF0, pN, _ = ts(ids, L, aln); return ds(pasr @ aln, pF0, pN).squeeze()
ref = [synth(r) for r in held]
def quantize(model, bits):
    nq = nbytes = 0
    for name, p in model.named_parameters():
        if p.dim() >= 2 and name.split(".")[-1].startswith("weight"):
            p.data = fq(p.data, bits); nq += p.numel(); nbytes += p.numel() * bits // 8 + p.shape[0] * 2   # + fp16 scale per channel
        else: nbytes += p.numel() * 2
    return nq, nbytes
qt, bt = quantize(ts, A.bits); qd, bd = quantize(ds, A.dec_bits)
out = [synth(r) for r in held]; raw, dtw, ld = compare(ref, out)
out_dir = P.out / "quant_student"; os.makedirs(out_dir, exist_ok=True)
for i, (o, r) in enumerate(zip(out, ref)):
    if i < 3:
        sf.write(f"{out_dir}/held_{i+1}_int{A.bits}_student.wav", o.numpy(), 24000); sf.write(f"{out_dir}/held_{i+1}_fp32_student.wav", r.numpy(), 24000)
        for name in (f"int{A.bits}", "fp32"): write_transcript(f"{out_dir}/held_{i+1}_{name}_student.wav", held[i]["text"])
tot = sum(p.numel() for p in ts.parameters()) + sum(p.numel() for p in ds.parameters())
print(f"params {tot/1e6:.2f}M | text side int{A.bits}: {bt/1e6:.2f} MB, decoder int{A.dec_bits}: {bd/1e6:.2f} MB, total {(bt+bd)/1e6:.2f} MB (fp32 would be {tot*4/1e6:.1f} MB)")
print(f"quantized vs fp32 student: DTW log-mel {dtw:.3f}, |dur| {ld:.0f} ms  (teacher seed-to-seed floor 0.09)")
