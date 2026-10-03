"""Tag works with the released student using Hugging Face transformers (one GPU; about 25 GB of GPU memory in bf16).
Input: JSON lines with id, title, abstract, venue (data/fetch_works.py writes them). Output: JSON lines
{id, top: [[topic id or NOT_CLASSIFIABLE, probability], ...]}, highest first.
Usage: python student/infer.py --model topic-classifier-v2 --input works.jsonl --out preds.jsonl.gz [--top 20]
For hundreds of millions of works use student/score_vllm.py (the production path, FP8)."""
import argparse, gzip, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CLASS_IDS, load, probabilities, work_text

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--top", type=int, default=20); ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--device", default="cuda"); ap.add_argument("--dtype", default="bfloat16")
a = ap.parse_args()
op = gzip.open if a.input.endswith(".gz") else open
rows = [json.loads(l) for l in op(a.input, "rt", encoding="utf-8") if l.strip()]
order = sorted(range(len(rows)), key=lambda i: len(rows[i].get("title") or "") + len(rows[i].get("abstract") or ""))   # less padding
tok, enc, head = load(a.model, a.device, a.dtype)
out = {}; t0 = time.time()
for s in range(0, len(order), a.batch):
    b = [rows[i] for i in order[s:s + a.batch]]
    P = probabilities(tok, enc, head, [work_text(r.get("title"), r.get("abstract"), r.get("venue")) for r in b])
    v, k = P.topk(a.top, 1)
    for r, vv, kk in zip(b, v.tolist(), k.tolist()):
        out[r["id"]] = [[CLASS_IDS[c], round(p, 6)] for c, p in zip(kk, vv)]
w = gzip.open(a.out, "wt") if a.out.endswith(".gz") else open(a.out, "w")
with w:
    for r in rows:
        w.write(json.dumps({"id": r["id"], "top": out[r["id"]]}) + "\n")
print(f"{len(rows):,} works in {time.time() - t0:.0f}s -> {a.out}")
