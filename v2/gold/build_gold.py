"""Build the panel gold from the judges' rows. One vote per judge per work: a topic id, "none" (not classifiable) or
missing (refused / error). Tiers:
  agreed        Opus 5.5 and Sol both voted and match                 -> gold = their answer
  tie-broken    they differ (or one is missing) and Fable matches one -> gold = the match ("two-vote" if one was missing)
  no-consensus  Fable matches neither
  unresolved    a needed vote is missing
An Opus 5.5 refusal is replaced by the Opus 5 row for that work (status opus5_fallback). acceptable_topic_ids = the gold
topic plus the secondary topics of the judges who chose it.
Usage: python gold/build_gold.py --split test [--judges data/gold/judges] [--out work/panel_gold_test.jsonl] [--need-fable work/need_fable.json]
With the defaults it rebuilds data/gold/panel_gold_<split>.jsonl from the released judge rows (it should be identical)."""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
from common import DATA, TOPIC, read_jsonl

ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True, choices=["dev", "test"])
ap.add_argument("--judges", default=os.path.join(DATA, "gold", "judges")); ap.add_argument("--out")
ap.add_argument("--need-fable", help="write the ids that still need a Fable vote here (JSON list, for gold/judge.py --ids-file)")
a = ap.parse_args(); S = a.split


def vote(ans):
    if ans is None: return None
    if ans.get("resolvable_level") == "none": return "none"
    t = ans.get("primary_topic_id"); return t if t in TOPIC else None


def load(judge):
    p = os.path.join(a.judges, f"{S}_{judge}.jsonl.gz"); d = {}
    if not os.path.exists(p): return d
    for r in read_jsonl(p):
        if r["status"] == "bad_topic_id" and (r.get("answer") or {}).get("resolvable_level") == "none": r["status"] = "none"
        if r["status"] in ("ok", "none"): d[r["id"]] = {"vote": vote(r["answer"]), "answer": r["answer"], "status": r["status"]}
        elif r["id"] not in d: d[r["id"]] = {"vote": None, "answer": None, "status": r["status"]}
    return d


works = [w for w in read_jsonl(os.path.join(DATA, "gold", f"{S}_works.jsonl")) if not w.get("missing")]
O, SO, O5 = load("opus"), load("sol"), load("opus5")
for i in [i for i, x in O.items() if x["status"] == "refused"]:
    if i in O5: O[i] = {**O5[i], "status": "opus5_fallback" if O5[i]["vote"] is not None else "opus5_" + O5[i]["status"]}
if S == "dev":   # the dev set's Fable vote is an earlier adjudicated Fable label (rows from Opus 5 are not Fable votes)
    F = {}
    for g in read_jsonl(os.path.join(a.judges, "dev_fable.jsonl.gz")):
        F[g["id"]] = ({"vote": None, "answer": None, "status": "opus5_fallback"} if g.get("fallback") else
                      {"vote": "none" if g.get("resolvable_level") == "none" else g["primary_topic_id"],
                       "answer": {"primary_topic_id": g["primary_topic_id"], "secondary_topic_ids": [t for t in g["topic_ids"] if t != g["primary_topic_id"]],
                                  "resolvable_level": g["resolvable_level"], "confidence": g.get("confidence")}, "status": "ok"})
else:
    F = load("fable")


def sec(x): return [t for t in ((x or {}).get("secondary_topic_ids") or []) if t in TOPIC]


out, tiers, need = [], collections.Counter(), []
for w in works:
    i = w["id"]; o, s, f = O.get(i), SO.get(i), F.get(i)
    if o is None or s is None: tiers["pending"] += 1; continue
    if o["vote"] is not None and o["vote"] == s["vote"]:
        tier, g, winners = "agreed", o["vote"], [o, s]
    else:
        if f is None: tiers["needs_fable"] += 1; need.append(i); continue
        votes = [x for x in (o, s) if x["vote"] is not None]
        if f["vote"] is None: tier, g, winners = "unresolved", None, []
        elif any(x["vote"] == f["vote"] for x in votes):
            tier = "tie-broken" if len(votes) == 2 else "two-vote"; g = f["vote"]; winners = [x for x in votes if x["vote"] == g] + [f]
        elif len(votes) < 2: tier, g, winners = "unresolved", None, []
        else: tier, g, winners = "no-consensus", None, []
    tiers[tier] += 1
    rec = {"id": i, "set": w["set"], "tier": tier, "primary_topic_id": g,
           "acceptable_topic_ids": sorted({g} | {t for x in winners for t in sec(x["answer"])} - {None, "none"}) if g not in (None, "none") else [],
           "votes": {"opus": o["vote"], "sol": s["vote"], "fable": (f or {}).get("vote")},
           "status": {"opus": o["status"], "sol": s["status"], "fable": (f or {}).get("status")}}
    if g not in (None, "none"):
        t = TOPIC[g]; rec.update(subfield_id=t["subfield"]["id"], field_id=t["field"]["id"], domain_id=t["domain"]["id"])
    out.append(rec)
path = a.out or os.path.join(DATA, "gold", f"panel_gold_{S}.jsonl")
with open(path, "w") as fh:
    for r in out: fh.write(json.dumps(r) + "\n")
print(S, dict(tiers), "->", path)
if a.need_fable: json.dump(need, open(a.need_fable, "w")); print(f"{len(need)} works need a Fable vote -> {a.need_fable}")
