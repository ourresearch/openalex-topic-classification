"""The teacher: a frontier model SELECTS one topic for a work from a short list of candidates (the topic vocabulary is
closed), or answers NONE_FIT (no candidate describes it) or NOT_CLASSIFIABLE (nothing to classify). It reads the title,
abstract and venue only. The released labels are Opus 5.5 at effort xhigh.
Run an arm on the gold works (rows look like data/teacher_arms/*):
  ANTHROPIC_API_KEY=... python teacher/teacher.py --teacher opus --effort xhigh --arm p02 --split test \
      --works work/test_texts.jsonl --out work/test_opus_xhigh_p02.jsonl
Arms: p01 / p02 / p03 / p05 = Jev shortlist at that threshold (data/shortlist); top60 = the retriever's top 60, no Jev.
Needs `pip install anthropic httpx` (httpx and OPENROUTER_API_KEY for the GPT models)."""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eval")); sys.path.insert(0, os.path.join(HERE, "..", "shortlist"))
from common import DATA, TOPIC, read_jsonl

PROMPT_VERSION = "select-v1-fixedschema"
SYSTEM = open(os.path.join(HERE, "prompt.md"), encoding="utf-8").read().split("## System prompt")[1].split("```\n", 1)[1].rsplit("\n```", 1)[0]
# One fixed schema for every request (per-request enums of candidate ids hit the API's grammar-compilation limit at
# scale); an id outside the shortlist is caught in code (status invalid_id).
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["topic_id", "secondary_topic_ids", "confidence"],
          "properties": {"topic_id": {"type": "string", "description": "a candidate ID, NONE_FIT or NOT_CLASSIFIABLE"},
                         "secondary_topic_ids": {"type": "array", "items": {"type": "string"}}, "confidence": {"type": "number"}}}
MODELS = {"opus": "claude-opus-5-5", "opus5": "claude-opus-5", "sol": "openai/gpt-6.1-sol", "astra": "openai/gpt-6-astra"}


def option(t): return f"{t['id']} | {t['display_name']} | {', '.join((t.get('keywords') or [])[:10])}"


def work_text(w):
    parts = [f"Title: {w.get('title') or '(none)'}"]
    if w.get("abstract"): parts.append(f"Abstract: {w['abstract'][:4000]}")
    parts.append(f"Venue: {w.get('venue') or '(unknown)'}")
    return "\n".join(parts)


def user(w, cids): return work_text(w) + "\n\nCandidate topics:\n" + "\n".join(option(TOPIC[c]) for c in cids)


def check(ans, cids):
    ok = ans.get("topic_id") in set(cids) | {"NONE_FIT", "NOT_CLASSIFIABLE"}
    ans["secondary_topic_ids"] = [x for x in ans.get("secondary_topic_ids") or [] if x in cids and x != ans.get("topic_id")][:2]
    return "ok" if ok else "invalid_id"


def request(w, cids, model="claude-opus-5-5", effort="xhigh"):
    """The exact Messages API request (as sent through the Batches API for the 2M labels)."""
    return {"model": model, "max_tokens": 16000, "system": SYSTEM, "messages": [{"role": "user", "content": user(w, cids)}],
            "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}}}


def ask_anthropic(client, w, cids, effort, model):
    try: r = client.messages.create(**request(w, cids, model, effort))
    except Exception as e: return {"status": "error", "error": repr(e)[:300]}
    rec = {"served_model": r.model, "stop_reason": r.stop_reason}
    if r.stop_reason == "refusal": return {**rec, "status": "refused"}
    if r.stop_reason == "max_tokens": return {**rec, "status": "max_tokens"}
    try: ans = json.loads(next(b.text for b in r.content if b.type == "text")); return {**rec, "status": check(ans, cids), "answer": ans}
    except Exception: return {**rec, "status": "bad_json"}


def ask_openrouter(w, cids, effort, model):
    import httpx
    body = {"model": model, "reasoning": {"effort": effort}, "max_output_tokens": 16000,
            "input": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user(w, cids)}],
            "text": {"format": {"type": "json_schema", "name": "topic", "schema": SCHEMA, "strict": True}}}
    err = None
    for k in range(6):
        try:
            d = httpx.post("https://openrouter.ai/api/v1/responses", json=body, timeout=600, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}).json()
            if d.get("error"): raise RuntimeError(str(d["error"])[:300])
            msg = next((o for o in d["output"] if o["type"] == "message"), None)
            if msg is None: return {"served_model": d.get("model"), "status": "no_message"}
            c0 = msg["content"][0]
            if c0.get("type") == "refusal": return {"served_model": d.get("model"), "status": "refused"}
            ans = json.loads(c0["text"]); return {"served_model": d.get("model"), "status": check(ans, cids), "answer": ans}
        except Exception as e: err = repr(e)[:300]; time.sleep(10 * (k + 1))
    return {"status": "error", "error": err}


def arm_shortlist(row, arm):
    from jev_shortlist import shortlist
    if arm == "top60": return row["candidates"][:60]
    return shortlist(row["candidates"], row["p"], {"p01": 0.01, "p02": 0.02, "p03": 0.03, "p05": 0.05}[arm])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, choices=list(MODELS)); ap.add_argument("--arm", required=True); ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--split", default="test", choices=["dev", "test"]); ap.add_argument("--works", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=12)
    a = ap.parse_args()
    J = {r["id"]: r for r in read_jsonl(os.path.join(DATA, "shortlist", f"{a.split}_jev_k255.jsonl.gz"))}
    done = {r["id"] for r in read_jsonl(a.out) if r["status"] in ("ok", "refused")} if os.path.exists(a.out) else set()
    works = [w for w in read_jsonl(a.works) if not w.get("missing") and w["id"] in J and w["id"] not in done]
    client = None
    if a.teacher in ("opus", "opus5"):
        import anthropic
        client = anthropic.Anthropic(max_retries=6, timeout=600)
    def one(w):
        cids = arm_shortlist(J[w["id"]], a.arm)
        r = ask_anthropic(client, w, cids, a.effort, MODELS[a.teacher]) if client else ask_openrouter(w, cids, a.effort, MODELS[a.teacher])
        return {"id": w["id"], "set": J[w["id"]].get("set"), "teacher": a.teacher, "effort": a.effort, "arm": a.arm, "n_cands": len(cids), "cands": cids,
                "prompt_version": PROMPT_VERSION, **r}
    with open(a.out, "a", encoding="utf-8") as out, ThreadPoolExecutor(a.concurrency) as ex:
        for f in as_completed([ex.submit(one, w) for w in works]):
            out.write(json.dumps(f.result(), ensure_ascii=False) + "\n"); out.flush()


if __name__ == "__main__":
    main()
