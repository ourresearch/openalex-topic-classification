"""How the second million was chosen: aim the teacher's labels at the topics the first million covered thinly.
Input: a pool of works scored by the small shortlist student (Qwen3-1.7B, release asset shortlist-student-qwen3-1.7b;
score with student/infer.py --top 10), and the first million's label counts (data/teacher_labels/label_counts.json).
Each pool work votes for its top topic (never NOT_CLASSIFIABLE); per topic keep the 2,000 most confident; choose the
level T so that sum_t min(available_t, max(0, T - labels_t)) is about 1M; take, for each topic, its most confident
T - labels_t works. The teacher then labels them with the student's top 10 as the shortlist.
In October 2026 the pool was 11.6M random works with a title of at least 10 characters (none from the first million or
the gold sets), T = 431, and 1,003,519 works were picked (data/teacher_labels/second_million_selection.json).
Usage: python teacher/select_tail.py --scored work/pool_scored.jsonl.gz --out work/selected.jsonl [--budget 1000000]"""
import argparse, heapq, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
from common import DATA, TOPIC_IDS, read_jsonl

ap = argparse.ArgumentParser(); ap.add_argument("--scored", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--budget", type=int, default=1_000_000); ap.add_argument("--keep", type=int, default=2000)
a = ap.parse_args()
c = {t: v["first_million"] for t, v in json.load(open(os.path.join(DATA, "teacher_labels", "label_counts.json"))).items()}
heaps = {}
for r in read_jsonl(a.scored):
    t, p = r["top"][0]
    if t == "NOT_CLASSIFIABLE": continue
    h = heaps.setdefault(t, []); item = (p, r["id"], [x[0] for x in r["top"] if x[0] != "NOT_CLASSIFIABLE"][:10])
    if len(h) < a.keep: heapq.heappush(h, item)
    elif item > h[0]: heapq.heapreplace(h, item)
avail = {t: sorted(h, reverse=True) for t, h in heaps.items()}
def total(T): return sum(min(len(avail.get(t, [])), max(0, T - c.get(t, 0))) for t in TOPIC_IDS)
lo, hi = 0, 100000
while lo < hi:
    mid = (lo + hi) // 2
    if total(mid) < a.budget: lo = mid + 1
    else: hi = mid
T = lo; n = 0
with open(a.out, "w") as f:
    for t in TOPIC_IDS:
        for p, wid, cands in avail.get(t, [])[:max(0, T - c.get(t, 0))]:
            f.write(json.dumps({"id": wid, "target_topic_id": t, "p": round(p, 4), "candidates": cands}) + "\n"); n += 1
print(f"T = {T}: {n:,} works -> {a.out}")
