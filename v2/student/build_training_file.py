"""Build the training file from the released teacher labels and the works' texts (tools/fetch_works.py).
Rows {id, ti, ab, ve, y, s, h}: y = class (topics.json index, 4516 = NOT_CLASSIFIABLE), s = the teacher's secondary
topics, h = held out (work id ending in 7). NONE_FIT labels (no candidate fit, 0.2%) are dropped. Order: the first
million, then the second, as in training (the trainer shuffles with a fixed seed).
Usage: python student/build_training_file.py --labels teacher_labels_2m.jsonl.gz --texts work/train_texts.jsonl --out work/train/t2m_train.jsonl.gz [--million 1]
Texts rebuilt from today's API will differ slightly from the September/October 2026 texts the student read."""
import argparse, gzip, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
from common import TOPIC_IDS, read_jsonl

ap = argparse.ArgumentParser(); ap.add_argument("--labels", required=True); ap.add_argument("--texts", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--million", type=int)
a = ap.parse_args()
IDX = {t: i for i, t in enumerate(TOPIC_IDS)}; IDX["NOT_CLASSIFIABLE"] = 4516
texts = {}
for w in read_jsonl(a.texts):
    if not w.get("missing"): texts[w["id"]] = (w.get("title") or "", w.get("abstract") or "", w.get("venue") or "")
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); n = skipped = 0
with gzip.open(a.out, "wt", encoding="utf-8") as out:
    for r in read_jsonl(a.labels):
        if (a.million and r["million"] != a.million) or r["status"] != "ok" or r["topic_id"] == "NONE_FIT": continue
        if r["id"] not in texts: skipped += 1; continue
        ti, ab, ve = texts[r["id"]]
        out.write(json.dumps({"id": r["id"], "ti": ti, "ab": ab, "ve": ve, "y": IDX[r["topic_id"]], "s": [IDX[x] for x in r.get("secondary_topic_ids") or [] if x in IDX],
                              "h": int(r["id"][1:]) % 10 == 7}, ensure_ascii=False) + "\n"); n += 1
print(f"{n:,} rows -> {a.out} ({skipped:,} labels without text)")
