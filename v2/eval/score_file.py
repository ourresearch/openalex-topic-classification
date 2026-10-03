"""Score any predictions file against the answer key, the same way as eval/test_read.py, and compare its top answers
with the released model's saved outputs.
Usage: python3 eval/score_file.py preds.jsonl.gz [--split test|dev]
Rows: {"id": ..., "top": [[topic id or NOT_CLASSIFIABLE, probability], ...]} (student/infer.py, student/score_vllm.py)."""
import sys
from common import *

path = sys.argv[1]; split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "test"
P = {r["id"]: r["top"] for r in read_jsonl(path)}
p = {i: t[0][0] for i, t in P.items()}; c = {i: t[0][1] for i, t in P.items()}
ids = [i for i in eval_ids(split) if i in P]
print(f"{len(P):,} predictions; {len(ids):,} of {len(eval_ids(split)):,} {split} works covered")
s = score(split, p, c); ra, ca, rt = s[("random", "agreed")], s[("cited", "agreed")], s[("random", "tie-broken")]
print("| file | random, agreed | field | calibration error | cited, agreed | random, tie-broken |\n|---|---|---|---|---|---|")
print(f"| {os.path.basename(path)} | {fmt(ra['acc'])} | {fmt(ra['field'])} | {fmt(ra['ece'])} | {fmt(ca['acc'])} | {fmt(rt['acc'])} |")
ref, _ = student(split, "q8b_2m")
print(f"same top answer as the released model's saved outputs: {sum(p[i] == ref.get(i) for i in ids):,} of {len(ids):,}")
