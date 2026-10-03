"""Parse batch results into labels: {id, status, topic_id, secondary_topic_ids, confidence, n_cands}. Status is ok,
invalid_id (an id outside the shortlist), refused, max_tokens, bad_json, errored or expired. Writes labels.jsonl,
retry_ids.json (anything but ok and refused) and refused_ids.json (for fallback.py).
Usage: python teacher/collect.py --dir work/req1"""
import argparse, collections, json, os
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args(); D = a.dir
cnt = collections.Counter(); retry, refused = [], []
with open(f"{D}/labels.jsonl", "w") as out:
    for f in sorted(os.listdir(f"{D}/res")):
        if not f.endswith(".jsonl"): continue
        meta = {m["id"]: m for m in map(json.loads, open(f"{D}/meta_{f[6:-6]}.jsonl"))}
        for r in map(json.loads, open(f"{D}/res/{f}")):
            m = meta[r["id"]]; rec = {"id": r["id"], "n_cands": len(m["cids"])}
            if r["type"] != "succeeded": st = r["type"]
            elif r["stop_reason"] == "refusal": st = "refused"
            elif r["stop_reason"] == "max_tokens": st = "max_tokens"
            else:
                try:
                    ans = json.loads(r["text"]); st = "ok" if ans.get("topic_id") in set(m["cids"]) | {"NONE_FIT", "NOT_CLASSIFIABLE"} else "invalid_id"
                    rec.update(topic_id=ans.get("topic_id"), confidence=ans.get("confidence"),
                               secondary_topic_ids=[x for x in ans.get("secondary_topic_ids") or [] if x in m["cids"] and x != ans.get("topic_id")][:2])
                except Exception: st = "bad_json"
            rec["status"] = st; cnt[st] += 1; out.write(json.dumps(rec) + "\n")
            if st == "refused": refused.append(r["id"])
            elif st != "ok": retry.append(r["id"])
json.dump(retry, open(f"{D}/retry_ids.json", "w")); json.dump(refused, open(f"{D}/refused_ids.json", "w"))
print(dict(cnt), f"retry {len(retry)} refused {len(refused)}")
