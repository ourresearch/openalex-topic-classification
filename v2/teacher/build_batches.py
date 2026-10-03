"""Write Message Batches API request files for the teacher: one request per work, 20,000 per file, with the shortlist
each work gets. The 2M released labels were made this way:
  first million:  shortlist = Jev p >= 0.02 over the retriever's top 255 (release asset jev_shortlists_first_million)
  second million: shortlist = the Qwen3-1.7B student's top 10 (select_tail.py), which tied Jev's on the dev set
The candidates each released label was chosen from are in teacher_labels_2m.jsonl.gz ("candidates"), so
  python teacher/build_batches.py --works work/texts.jsonl --labels teacher_labels_2m.jsonl.gz --million 1 --out work/req1
rebuilds the exact requests (given the same texts). Next: submit_batches.py, collect.py, fallback.py."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher import request, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--works", required=True); ap.add_argument("--labels", required=True, help="rows with id + candidates")
ap.add_argument("--million", type=int); ap.add_argument("--out", required=True); ap.add_argument("--per-file", type=int, default=20000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
C = {r["id"]: r["candidates"] for r in read_jsonl(a.labels) if a.million is None or r.get("million") == a.million}
n = 0; f = m = None
for w in read_jsonl(a.works):
    if w.get("missing") or w["id"] not in C: continue
    if n % a.per_file == 0:
        if f: f.close(); m.close()
        k = n // a.per_file; f = open(f"{a.out}/batch_{k:03d}.jsonl", "w"); m = open(f"{a.out}/meta_{k:03d}.jsonl", "w")
    f.write(json.dumps({"custom_id": w["id"], "params": request(w, C[w["id"]])}, ensure_ascii=False) + "\n")
    m.write(json.dumps({"id": w["id"], "cids": C[w["id"]]}) + "\n"); n += 1
if f: f.close(); m.close()
print(f"{n:,} requests -> {a.out}")
