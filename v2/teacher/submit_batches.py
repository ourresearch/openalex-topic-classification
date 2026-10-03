"""Submit request files to the Message Batches API, at most MAX_INFLIGHT queued at once; poll; stream each ended batch's
results to <dir>/res/batch_NNN.jsonl ({id, type, stop_reason, model, in, out, text}). Resumable (state.json).
Usage: ANTHROPIC_API_KEY=... python teacher/submit_batches.py --dir work/req1 [--max-inflight 10]"""
import argparse, json, os, time, anthropic

ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--max-inflight", type=int, default=10)
a = ap.parse_args()
RES, ST = f"{a.dir}/res", f"{a.dir}/state.json"; os.makedirs(RES, exist_ok=True)
client = anthropic.Anthropic(max_retries=6, timeout=1800)
state = json.load(open(ST)) if os.path.exists(ST) else {}
def save(): json.dump(state, open(ST + ".tmp", "w"), indent=1); os.replace(ST + ".tmp", ST)
def log(*x): print(time.strftime("[%H:%M:%S]"), *x, flush=True)
names = sorted(f[6:-6] for f in os.listdir(a.dir) if f.startswith("batch_") and f.endswith(".jsonl"))
def collect(nn):
    tmp = f"{RES}/batch_{nn}.jsonl.tmp"; n = 0
    with open(tmp, "w") as out:
        for r in client.messages.batches.results(state[nn]["batch_id"]):
            rec = {"id": r.custom_id, "type": r.result.type}
            if r.result.type == "succeeded":
                msg = r.result.message
                rec.update(stop_reason=msg.stop_reason, model=msg.model, text=next((b.text for b in msg.content if b.type == "text"), None),
                           **{"in": msg.usage.input_tokens, "out": msg.usage.output_tokens})
            elif r.result.type == "errored": rec["error"] = str(r.result.error)[:300]
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
    os.replace(tmp, f"{RES}/batch_{nn}.jsonl"); state[nn]["collected"] = n; save(); log(f"batch {nn}: collected {n}")
while True:
    for nn, s in state.items():
        if s.get("status") != "ended":
            b = client.messages.batches.retrieve(s["batch_id"])
            if b.processing_status == "ended": s.update(status="ended", counts=b.request_counts.to_dict()); save(); log(f"batch {nn} ended", s["counts"])
        if s.get("status") == "ended" and "collected" not in s: collect(nn)
    inflight = [nn for nn, s in state.items() if s.get("status") != "ended"]; todo = [nn for nn in names if nn not in state]
    while todo and len(inflight) < a.max_inflight:
        nn = todo.pop(0); reqs = [json.loads(l) for l in open(f"{a.dir}/batch_{nn}.jsonl")]
        try: b = client.messages.batches.create(requests=reqs)
        except anthropic.APIStatusError as e: log(f"batch {nn} create failed {e.status_code}; retry next round"); break
        state[nn] = {"batch_id": b.id, "status": b.processing_status}; save(); inflight.append(nn); log(f"batch {nn} submitted {b.id}")
    if not todo and len(state) == len(names) and all("collected" in s for s in state.values()): log("all batches collected"); break
    time.sleep(120)
