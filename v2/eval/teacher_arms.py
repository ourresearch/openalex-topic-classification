"""Choosing the teacher: every teacher arm on the development set, then the chosen arms on the test set, against the
panel gold, with paired (McNemar) comparisons. Arms are named <model>_<effort>_<shortlist>: p01/p02/p03/p05 = every
candidate Jev gives at least that probability out of the retriever's top 255 (at least 2, at most 30); top60 = the
retriever's top 60 with no Jev; q17top10 = a small student's top 10. Refusals count as wrong here.
Usage: python3 eval/teacher_arms.py"""
import os
from common import *

NAMES = {"opus": "Opus 5.5", "opus5": "Opus 5", "sol": "GPT-6.1 Sol", "astra": "GPT-6 Astra"}
def label(arm):
    m, e, s = arm.split("_"); return f"{NAMES[m]} {e}, {s}"
for split in ("dev", "test"):
    arms = sorted(f[:-9] for f in os.listdir(os.path.join(DATA, "teacher_arms", split)))
    print(f"## {split}\n\n| arm | random, agreed | field | cited, agreed | random, tie-broken | refused | mean options |\n|---|---|---|---|---|---|---|")
    P = {}
    jp, jc = jev_alone(split); s = score(split, jp, jc)
    print(f"| Jev alone, 255 candidates | {fmt(s[('random','agreed')]['acc'])} | {fmt(s[('random','agreed')]['field'])} | {fmt(s[('cited','agreed')]['acc'])} | {fmt(s[('random','tie-broken')]['acc'])} | 0 | 255 |")
    for arm in arms:
        p, c, rows = teacher_arm(split, arm); P[arm] = p; s = score(split, p, c)
        ref = sum(r["status"] == "refused" for r in rows.values()) / len(rows); opts = sum(r.get("n_cands", 0) for r in rows.values()) / len(rows)
        print(f"| {label(arm)} | {fmt(s[('random','agreed')]['acc'])} | {fmt(s[('random','agreed')]['field'])} | {fmt(s[('cited','agreed')]['acc'])} | "
              f"{fmt(s[('random','tie-broken')]['acc'])} | {ref:.1%} | {opts:.1f} |")
    print("\nPaired on random works, agreed tier (only A right / only B right, z):\n")
    pairs = [("opus_xhigh_p02", "opus_medium_p02"), ("opus_xhigh_p02", "sol_medium_p02"), ("opus_medium_p02", "sol_medium_p02"),
             ("opus_xhigh_p02", "opus_high_p02"), ("opus_xhigh_p02", "astra_high_p02"), ("opus_xhigh_p02", "opus5_xhigh_p02"),
             ("opus_medium_p02", "opus_medium_top60"), ("opus_medium_p02", "opus_medium_p05"), ("opus_xhigh_q17top10", "opus_xhigh_p02")]
    for a, b in pairs:
        if a in P and b in P:
            oa, ob, z = mcnemar(split, P[a], P[b]); print(f"- {label(a)} vs {label(b)}: {oa} vs {ob}, z = {z:.1f}")
    print()
