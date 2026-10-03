"""The panel judges: a frontier model reads one work and picks its topic from the WHOLE taxonomy (all 4,516 topics in a
cached system prompt), blind to every other model. Opus 5.5 (xhigh) and GPT-6.1 Sol (high) label every gold work; Fable
5.1 (xhigh) labels the works where they differ; an Opus 5.5 refusal is re-asked of Opus 5 ("opus5"), flagged.
A "none" answer (resolvable_level none: the record has nothing to classify) is a valid vote.
Usage: ANTHROPIC_API_KEY=... OPENROUTER_API_KEY=... python gold/judge.py --judge opus|sol|fable|opus5 \
         --works work/test_texts.jsonl --gold-works data/gold/test_works.jsonl --out work/test_opus.jsonl [--ids-file f]
Resumable. Needs `pip install anthropic httpx`. Rows look like data/gold/judges/*.jsonl.gz."""
import argparse, collections, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
from common import TOPICS, TOPIC, read_jsonl

PROMPT_VERSION = "judge-pass1-seed0"
MODELS = {"opus": "claude-opus-5-5", "opus5": "claude-opus-5", "fable": "claude-fable-5-1", "sol": "openai/gpt-6.1-sol"}
EFFORT = {"opus": "xhigh", "opus5": "xhigh", "fable": "xhigh", "sol": "high"}

SYSTEM_HEAD = """You are building a gold-standard dataset of research-topic labels for OpenAlex, a scholarly index of hundreds of millions of works. Below is OpenAlex's complete topic taxonomy: 4 domains, 26 fields, 252 subfields, 4,516 topics. Each topic line is `ID | name | keywords`.

For each work you are given (title, abstract if any, venue, year, type, language), decide which topic best describes what the work is ABOUT. Rules:
- Read the work first, then find the best domain, field, subfield and topic. Consider several candidate topics before choosing; near-sibling topics differ in emphasis, so use the keywords.
- `primary_topic_id` must be the single best topic. `secondary_topic_ids` are up to 2 other topics that also genuinely fit (empty if none do).
- `domain_id`, `field_id`, `subfield_id` must be the ancestors of `primary_topic_id`.
- `resolvable_level` is the deepest level you can assign with real confidence from the available text: "topic" normally; "subfield"/"field"/"domain" when the text only supports a coarser call (still fill in your best-guess topic path); "none" when the record is not classifiable at all (empty or garbage title, pure boilerplate, or the taxonomy has no place for it). Datasets, software, books, theses and non-English works get a topic like anything else, based on their subject matter.
- Non-English text: read it in its own language; do not default to a catch-all topic.
- `confidence` is your probability (0-1) that `primary_topic_id` is the topic a careful domain expert would pick from this taxonomy.
- `reason`: one sentence, under 40 words, naming the decisive evidence.
Output only the JSON object.
"""

SCHEMA = {"type": "object", "properties": {
    "domain_id": {"type": "string"}, "field_id": {"type": "string"}, "subfield_id": {"type": "string"},
    "primary_topic_id": {"type": "string"},
    "secondary_topic_ids": {"type": "array", "items": {"type": "string"}},
    "resolvable_level": {"type": "string", "enum": ["topic", "subfield", "field", "domain", "none"]},
    "confidence": {"type": "number"}, "reason": {"type": "string"}},
    "required": ["domain_id", "field_id", "subfield_id", "primary_topic_id", "secondary_topic_ids", "resolvable_level", "confidence", "reason"],
    "additionalProperties": False}


def taxonomy_text():
    """The taxonomy as an outline, in topics.json order within each subfield."""
    tree = collections.OrderedDict()
    for t in TOPICS:
        tree.setdefault((t["domain"]["id"], t["domain"]["display_name"]), collections.OrderedDict()) \
            .setdefault((t["field"]["id"], t["field"]["display_name"]), collections.OrderedDict()) \
            .setdefault((t["subfield"]["id"], t["subfield"]["display_name"]), []).append(t)
    lines = []
    for (did, dn), fields in tree.items():
        lines.append(f"\n# DOMAIN {did}: {dn}")
        for (fid, fn), subs in fields.items():
            lines.append(f"\n## FIELD {fid}: {fn}")
            for (sid, sn), topics in subs.items():
                lines.append(f"\n### SUBFIELD {sid}: {sn}")
                for t in topics:
                    lines.append(f"- {t['id']} | {t['display_name']} | {', '.join((t.get('keywords') or [])[:12])}")
    return "\n".join(lines)


SYSTEM = SYSTEM_HEAD + "\n\n# TAXONOMY\n" + taxonomy_text()


def work_text(w):
    return "\n".join([f"Title: {w.get('title') or '(none)'}", f"Abstract: {w['abstract'][:4000] if w.get('abstract') else '(none)'}",
                      f"Venue: {w.get('venue') or '(unknown)'}",
                      f"Year: {w.get('year')}; Type: {w.get('type')}; Language: {w.get('language') or 'unknown'}"])


def finish(ans):
    """-> status, answer, flags. A 'none' answer with no valid topic is a valid vote; ancestors are taken from the topic."""
    t = TOPIC.get(ans.get("primary_topic_id"))
    if ans.get("resolvable_level") == "none" and not t:
        return {"status": "none", "answer": ans, "flags": []}
    if not t:
        return {"status": "bad_topic_id", "answer": ans, "flags": ["bad_topic_id"]}
    flags = [f"{k}_mismatch" for k in ("subfield", "field", "domain") if t[k]["id"] != ans.get(f"{k}_id")]
    ans.update(subfield_id=t["subfield"]["id"], field_id=t["field"]["id"], domain_id=t["domain"]["id"])
    ans["secondary_topic_ids"] = [s for s in ans.get("secondary_topic_ids") or [] if s in TOPIC and s != t["id"]][:2]
    return {"status": "ok", "answer": ans, "flags": flags}


def ask_anthropic(client, model, effort, w):
    r = client.messages.create(model=model, max_tokens=32000, system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                               messages=[{"role": "user", "content": "Classify this work.\n\n" + work_text(w)}],
                               output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}})
    rec = {"served_model": r.model, "stop_reason": r.stop_reason}
    if r.stop_reason == "refusal": return {**rec, "status": "refused"}
    if r.stop_reason == "max_tokens": return {**rec, "status": "max_tokens"}
    try: return {**rec, **finish(json.loads(next(b.text for b in r.content if b.type == "text")))}
    except Exception: return {**rec, "status": "bad_json"}


def ask_openrouter(model, effort, w):
    import httpx
    body = {"model": model, "reasoning": {"effort": effort}, "max_output_tokens": 32000,
            "input": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Classify this work.\n\n" + work_text(w)}],
            "text": {"format": {"type": "json_schema", "name": "topic", "schema": SCHEMA, "strict": True}}}
    err = None
    for k in range(6):
        try:
            d = httpx.post("https://openrouter.ai/api/v1/responses", json=body, timeout=900, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}).json()
            if d.get("error"): raise RuntimeError(str(d["error"])[:300])
            msg = next((o for o in d["output"] if o["type"] == "message"), None)
            if msg is None: return {"served_model": d.get("model"), "status": "no_message"}
            c0 = msg["content"][0]
            if c0.get("type") == "refusal": return {"served_model": d.get("model"), "status": "refused"}
            return {"served_model": d.get("model"), **finish(json.loads(c0["text"]))}
        except Exception as e:
            err = repr(e)[:300]; time.sleep(10 * (k + 1))
    return {"status": "error", "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=list(MODELS)); ap.add_argument("--works", required=True)
    ap.add_argument("--gold-works", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ids-file")
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    sets = {r["id"]: r["set"] for r in read_jsonl(a.gold_works)}
    done = {r["id"] for r in read_jsonl(a.out) if r["status"] in ("ok", "refused", "none")} if os.path.exists(a.out) else set()
    works = [w for w in read_jsonl(a.works) if not w.get("missing") and w["id"] in sets and w["id"] not in done]
    if a.ids_file: keep = set(json.load(open(a.ids_file))); works = [w for w in works if w["id"] in keep]
    client = None
    if a.judge != "sol":
        import anthropic
        client = anthropic.Anthropic(max_retries=6, timeout=1200)
    def one(w):
        try: r = ask_openrouter(MODELS["sol"], EFFORT["sol"], w) if a.judge == "sol" else ask_anthropic(client, MODELS[a.judge], EFFORT[a.judge], w)
        except Exception as e: r = {"status": "error", "error": repr(e)[:300]}
        return {"id": w["id"], "set": sets[w["id"]], "judge": a.judge, "model": MODELS[a.judge], "effort": EFFORT[a.judge], "prompt_version": PROMPT_VERSION, **r}
    print(f"{a.judge}: {len(works):,} works to judge", flush=True)
    import gzip
    with (gzip.open(a.out, "at", encoding="utf-8") if a.out.endswith(".gz") else open(a.out, "a", encoding="utf-8")) as out:
        if works: out.write(json.dumps(one(works[0]), ensure_ascii=False) + "\n"); works = works[1:]   # warm the prompt cache once
        with ThreadPoolExecutor(a.concurrency) as ex:
            for f in as_completed([ex.submit(one, w) for w in works]):
                out.write(json.dumps(f.result(), ensure_ascii=False) + "\n"); out.flush()


if __name__ == "__main__":
    main()
