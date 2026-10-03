"""The panel gold: how the answer key was built and how often the judges agree (standard library).
Opus 5.5 (xhigh) and GPT-6.1 Sol (high) label every work blind from the whole 4,516-topic taxonomy; where they agree,
that is the gold ("agreed"); where they differ, Fable 5.1 votes blind and its match wins ("tie-broken"); a three-way
split is "no-consensus". On the development set, Fable's vote is an earlier adjudicated label. Opus 5.5 refusals
(biomedical papers) are re-asked of Opus 5 and flagged.
Usage: python3 eval/panel.py"""
import collections
from common import *

for split in ("dev", "test"):
    G = gold(split)
    print(f"## {split}\n")
    print("| set | works | agreed | tie-broken | no-consensus | unresolved | judges agree: topic | subfield | field |")
    print("|---|---|---|---|---|---|---|---|---|")
    for st in ("random", "cited"):
        R = [g for g in G.values() if g["set"] == st]; c = collections.Counter(g["tier"] for g in R)
        both = [g for g in R if g["votes"]["opus"] is not None and g["votes"]["sol"] is not None]
        def lvl(t, k): return None if t in (None, "none") else TOPIC[t][k]["id"]
        agree = {k: sum((g["votes"]["opus"] == g["votes"]["sol"]) if k == "topic" else (lvl(g["votes"]["opus"], k) == lvl(g["votes"]["sol"], k)) for g in both) / max(1, len(both))
                 for k in ("topic", "subfield", "field")}
        print(f"| {st} | {len(R):,} | {c['agreed']:,} | {c['tie-broken'] + c['two-vote']:,} | {c['no-consensus']:,} | {c['unresolved']:,} | "
              f"{agree['topic']:.3f} | {agree['subfield']:.3f} | {agree['field']:.3f} |")
    none = sum(1 for g in G.values() if g["tier"] == "agreed" and g["primary_topic_id"] == "none")
    fb = sum(1 for g in G.values() if g["status"]["opus"] == "opus5_fallback")
    print(f"\nagreed 'not classifiable': {none}; Opus seat filled by Opus 5 after an Opus 5.5 refusal: {fb}\n")
