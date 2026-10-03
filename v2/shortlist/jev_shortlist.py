"""Jev scores the retriever's top 255 topics for a work (one request per work) and returns a probability for each; the
teacher's shortlist is every candidate with p >= 0.02 (at least 2, at most 30, in Jev's order). Jev is a decision model
from TypeSafe AI (https://typesafe.ai); its probabilities come in steps of 0.01.
Usage: JEV_API_KEY=... python shortlist/jev_shortlist.py --works work/texts.jsonl --cands work/cands255.jsonl --out work/jev.jsonl
The request below is exactly the one used; `client.decide` stands for one call to Jev's decide endpoint with a state
text and one choice question (see Jev's documentation for the HTTP form). Rows look like data/shortlist/*.jsonl.gz."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
from common import TOPIC, read_jsonl

INSTRUCTIONS = "Which research topic best describes what this work is about?"
TAU, FLOOR, CAP = 0.02, 2, 30


def option_text(t): return f"{t['display_name']}: {', '.join((t.get('keywords') or [])[:10])}"


def state_text(w):
    parts = [f"Title: {w.get('title') or ''}"]
    if w.get("abstract"): parts.append(f"Abstract: {w['abstract'][:3000]}")
    if w.get("venue"): parts.append(f"Venue: {w['venue']}")
    return "\n".join(parts)


def request(w, candidates):
    """The Jev request: state = the work, one choice over the candidates (in retriever order)."""
    return {"state": state_text(w), "questions": {"t": {"type": "choice", "instructions": INSTRUCTIONS,
                                                        "criteria": {c: option_text(TOPIC[c]) for c in candidates}}}}


def shortlist(candidates, p, tau=TAU):
    """The teacher's shortlist: candidates with p >= tau, highest first (ties in retriever order), at least 2, at most 30."""
    order = sorted(range(len(candidates)), key=lambda k: -p[k])   # stable: ties keep retriever order
    n = sum(1 for x in p if x >= tau)
    return [candidates[k] for k in order[:min(max(n, FLOOR), CAP)]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--works", required=True); ap.add_argument("--cands", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from jev_client import decide   # your own thin client: decide(state, questions) -> {"t": {"probabilities": {topic id: p}}}
    C = {r["id"]: r["candidates"] for r in read_jsonl(a.cands)}
    with open(a.out, "a") as f:
        for w in read_jsonl(a.works):
            if w["id"] not in C: continue
            req = request(w, C[w["id"]]); ans = decide(req["state"], req["questions"])["t"]["probabilities"]
            f.write(json.dumps({"id": w["id"], "candidates": C[w["id"]], "p": [round(float(ans.get(c, 0.0)), 5) for c in C[w["id"]]]}) + "\n")


if __name__ == "__main__":
    main()
