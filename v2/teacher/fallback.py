"""The fallback chain for works the teacher did not label (about 1%, mostly biomedical papers Opus 5.5 refuses):
  1. retry_ids    -> Opus 5.5 xhigh again (live calls, same request)
  2. refusals     -> Opus 5 xhigh (more accurate than GPT-6 Astra on the dev set, 85.5% vs 82.9%, but it refuses ~9% of them)
  3. still none   -> GPT-6 Astra high via OpenRouter (never refused in our runs)
then merge into labels_final.jsonl with each label's source model. Every one of the 2M works ended with a label.
Usage: ANTHROPIC_API_KEY=... OPENROUTER_API_KEY=... python teacher/fallback.py --dir work/req1"""
import argparse, collections, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anthropic
from teacher import SYSTEM, SCHEMA, check

ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--threads", type=int, default=48)
a = ap.parse_args(); D = a.dir
client = anthropic.Anthropic(max_retries=6, timeout=1800)
retry, refused = json.load(open(f"{D}/retry_ids.json")), json.load(open(f"{D}/refused_ids.json")); want = set(retry) | set(refused)
REQ, CIDS = {}, {}
for f in sorted(os.listdir(D)):
    if f.startswith("meta_"): CIDS.update({m["id"]: m["cids"] for m in map(json.loads, open(f"{D}/{f}")) if m["id"] in want})
    elif f.startswith("batch_") and f.endswith(".jsonl"):
        for l in open(f"{D}/{f}"):
            r = json.loads(l)
            if r["custom_id"] in want: REQ[r["custom_id"]] = r["params"]
def live(i, model):
    try: m = client.messages.create(**{**REQ[i], "model": model})
    except Exception as e: return {"id": i, "status": "error", "error": repr(e)[:200]}
    if m.stop_reason == "refusal": return {"id": i, "status": "refused", "model": model}
    try: ans = json.loads(next(b.text for b in m.content if b.type == "text")); return {"id": i, "status": check(ans, CIDS[i]), "answer": ans, "model": model}
    except Exception: return {"id": i, "status": "bad_json", "model": model}
with ThreadPoolExecutor(a.threads) as ex: R1 = {r["id"]: r for r in ex.map(lambda i: live(i, "claude-opus-5-5"), retry)}
second = list(refused) + [i for i, r in R1.items() if r["status"] != "ok"]
with ThreadPoolExecutor(a.threads) as ex: R2 = {r["id"]: r for r in ex.map(lambda i: live(i, "claude-opus-5"), second)}
got = {i: r for i, r in R1.items() if r["status"] == "ok"}; got.update({i: r for i, r in R2.items() if r["status"] == "ok"})
need = [i for i in want if i not in got]
def astra(i):
    import httpx
    w_cids = CIDS[i]; body_user = REQ[i]["messages"][0]["content"]
    body = {"model": "openai/gpt-6-astra", "reasoning": {"effort": "high"}, "max_output_tokens": 16000,
            "input": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body_user}],
            "text": {"format": {"type": "json_schema", "name": "topic", "schema": SCHEMA, "strict": True}}}
    for k in range(5):
        try:
            d = httpx.post("https://openrouter.ai/api/v1/responses", json=body, timeout=300, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}).json()
            msg = next(o for o in d["output"] if o["type"] == "message"); c0 = msg["content"][0]
            if c0.get("type") == "refusal": return {"id": i, "status": "refused", "model": "gpt-6-astra"}
            ans = json.loads(c0["text"]); return {"id": i, "status": check(ans, w_cids), "answer": ans, "model": "gpt-6-astra"}
        except Exception: time.sleep(5 * (k + 1))
    return {"id": i, "status": "error"}
with ThreadPoolExecutor(12) as ex: got.update({r["id"]: r for r in ex.map(astra, need) if r["status"] == "ok"})
cnt = collections.Counter()
with open(f"{D}/labels_final.jsonl", "w") as out:
    for r in map(json.loads, open(f"{D}/labels.jsonl")):
        if r["status"] == "ok": r["model"] = "claude-opus-5-5"
        elif r["id"] in got:
            g = got[r["id"]]; ans = g["answer"]
            r = {"id": r["id"], "n_cands": r["n_cands"], "status": "ok", "model": g["model"], "topic_id": ans["topic_id"], "confidence": ans.get("confidence"),
                 "secondary_topic_ids": [x for x in ans.get("secondary_topic_ids") or [] if x in CIDS[r["id"]] and x != ans["topic_id"]][:2]}
        cnt[r.get("model") if r["status"] == "ok" else "unlabelled"] += 1; out.write(json.dumps(r) + "\n")
print(dict(cnt))
