"""Rollout evidence on the test set (standard library): how far each model's confidence can be trusted, and what the
student does with records that have nothing to classify.
Usage: python3 eval/confidence.py"""
from common import *

G = gold("test"); ids = [i for i in eval_ids("test") if i in G and G[i]["set"] == "random" and G[i]["tier"] == "agreed"]
sp, sc = student("test", "q8b_2m"); op, oc = old_model("test")
def right(p, i): return p.get(i) == gold_answer(G[i])
print("## The student's probability, random works, agreed tier\n\n| band | share of works | accuracy |\n|---|---|---|")
for lo, hi in ((0.9, 1.01), (0.7, 0.9), (0.5, 0.7), (0.0, 0.5)):
    ii = [i for i in ids if lo <= sc[i] < hi]
    print(f"| {lo:.1f} to {min(hi, 1):.1f} | {len(ii) / len(ids):.0%} | {sum(right(sp, i) for i in ii) / max(1, len(ii)):.3f} |")
print("\n## Where the previous model was confident, random works, agreed tier\n\n| previous model's score | works | previous model right | student right |\n|---|---|---|---|")
for t in (0.5, 0.9, 0.95):
    ii = [i for i in ids if (oc.get(i) or 0) >= t and op.get(i)]
    print(f"| at least {t} | {len(ii):,} | {sum(right(op, i) for i in ii) / len(ii):.3f} | {sum(right(sp, i) for i in ii) / len(ii):.3f} |")
nc = [i for i in ids if sp.get(i) == NC_LABEL]
print(f"\n## Not classifiable\n\nThe student says not classifiable on {len(nc)} of {len(ids):,} random agreed works ({len(nc) / len(ids):.0%}); "
      f"the panel agrees on {sum(gold_answer(G[i]) == NC_LABEL for i in nc)} of them. "
      f"The panel says not classifiable on {sum(gold_answer(G[i]) == NC_LABEL for i in ids)} works.")
nt = sum(1 for i in ids if op.get(i) is None)
print(f"The previous model gave no topic at all to {nt} of the {len(ids):,}.")
