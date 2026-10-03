"""Choosing the student, on the development set and on held-out training rows (standard library).
Teacher fidelity = how often the student gives the teacher's answer on 20,000 held-out training works (work id ending
in 7, never trained on); it separates 1-point differences the 766-work gold cannot. Bands = how many training labels
the teacher's answer had in the first million (held-out rows not counted; NOT_CLASSIFIABLE falls in the top band).
Usage: python3 eval/students.py"""
import json, os
from common import *

ROWS = [("e5_full_e3", "multilingual-e5-base classifier (retriever init), 1M labels"), ("q17_e1", "Qwen3-1.7B, 1M labels"),
        ("q4b_e1", "Qwen3-4B, 1M labels"), ("q8b_e1", "Qwen3-8B, 1M labels"), ("q17_2m", "Qwen3-1.7B, 2M labels"),
        ("q8b_2m", "Qwen3-8B, 2M labels (released)"),
        ("x_soft", "1.7B: soft labels (0.2 on secondaries, 0.05 smoothing)"), ("x_mean", "1.7B: mean pooling"),
        ("x_len512", "1.7B: 512 tokens"), ("x_ep2", "1.7B: 2 epochs"), ("x_half", "1.7B: half the labels"),
        ("x_head2", "1.7B: head initialised from topic texts"), ("x_aux", "1.7B: subfield + field auxiliary loss"), ("x_lr4", "1.7B: learning rate 4e-5")]
counts = json.load(open(os.path.join(DATA, "teacher_labels", "label_counts.json")))
BANDS = [("< 50", 0, 50), ("50-200", 50, 200), ("200-1K", 200, 1000), ("> 1K", 1000, 10 ** 9)]
def band(t):
    if t not in counts: return None
    n = counts[t]["first_million_training"]; return next(b for b, lo, hi in BANDS if lo <= n < hi)
print("| student | teacher fidelity | " + " | ".join(f"fidelity, topics with {b} labels" for b, _, _ in BANDS) + " | dev random agreed | field | calibration error | dev cited agreed |")
print("|---" * (6 + len(BANDS)) + "|")
for tag, name in ROWS:
    H = list(read_jsonl(os.path.join(DATA, "student_outputs", "heldout", f"{tag}.jsonl.gz")))
    fid = sum(r["student"] == r["teacher"] for r in H) / len(H); byb = {}
    for b, _, _ in BANDS:
        rr = [r for r in H if band(r["teacher"]) == b]; byb[b] = sum(r["student"] == r["teacher"] for r in rr) / max(1, len(rr))
    p, c = student("dev", tag); s = score("dev", p, c)
    print(f"| {name} | {fid:.3f} | " + " | ".join(f"{byb[b]:.3f}" for b, _, _ in BANDS) +
          f" | {fmt(s[('random','agreed')]['acc'])} | {fmt(s[('random','agreed')]['field'])} | {fmt(s[('random','agreed')]['ece'])} | {fmt(s[('cited','agreed')]['acc'])} |")
H = list(read_jsonl(os.path.join(DATA, "student_outputs", "heldout", "q8b_2m.jsonl.gz")))
print("\nheld-out works per band: " + ", ".join(f"{b} {sum(1 for r in H if band(r['teacher']) == b):,}" for b, _, _ in BANDS))
a, b = student("dev", "q8b_2m")[0], student("dev", "q8b_e1")[0]; oa, ob, z = mcnemar("dev", a, b)
print(f"paired, dev random agreed: 8B on 2M vs 8B on 1M {oa} vs {ob}, z = {z:.1f}")
