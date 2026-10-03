"""Shared loading and scoring for every table in the benchmarks. Standard library only.

Classes: topics.json order (class i = topic i), plus NOT_CLASSIFIABLE (the record has no subject matter to classify).
Accuracy is always against the panel gold (two frontier models labelling blind from the whole taxonomy; a third breaks
ties), reported by set (random, cited) and tier (agreed, tie-broken); random and cited are never pooled."""
import gzip, json, math, os, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
NC_LABEL = "NOT_CLASSIFIABLE"


def read_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


TOPICS = json.load(open(os.path.join(DATA, "topics", "topics.json"), encoding="utf-8"))
TOPIC = {t["id"]: t for t in TOPICS}
TOPIC_IDS = [t["id"] for t in TOPICS]


def field_of(answer):
    """Field id of an answer; NOT_CLASSIFIABLE and "no answer" both map to "none" (as in the original scorer)."""
    t = TOPIC.get(answer)
    return t["field"]["id"] if t else "none"


def gold(split):
    return {r["id"]: r for r in read_jsonl(os.path.join(DATA, "gold", f"panel_gold_{split}.jsonl"))}


def gold_answer(g):
    """The gold primary topic, or NOT_CLASSIFIABLE when the panel agreed the record is not classifiable."""
    return NC_LABEL if g["primary_topic_id"] == "none" else g["primary_topic_id"]


def eval_ids(split):
    """The works every method is scored on: gold works with a record (shortlist rows exist for exactly these)."""
    return [r["id"] for r in read_jsonl(os.path.join(DATA, "shortlist", f"{split}_jev_k255.jsonl.gz"))]


def ece(conf, ok, bins=15):
    """Expected calibration error of the top answer's probability, 15 equal-width bins."""
    n = len(conf); e = 0.0
    for b in range(bins):
        lo = b * (1.0 / bins); hi = lo + 1.0 / bins   # the same float edges as numpy.linspace (confidences like 0.6 sit on edges)
        idx = [i for i in range(n) if lo < conf[i] <= hi]
        if idx:
            e += len(idx) / n * abs(sum(conf[i] for i in idx) / len(idx) - sum(ok[i] for i in idx) / len(idx))
    return e


def score(split, pred, conf=None):
    """pred: {work id: topic id | NOT_CLASSIFIABLE | None (no answer)}; conf: {work id: probability}.
    Returns {(set, tier): {"n", "acc", "field", "ece"}} over the eval ids. A missing answer is wrong."""
    G = gold(split); ids = eval_ids(split); out = {}
    for st in ("random", "cited"):
        for tier in ("agreed", "tie-broken"):
            ii = [i for i in ids if i in G and G[i]["set"] == st and G[i]["tier"] == tier]
            ok = [int(pred.get(i) == gold_answer(G[i])) for i in ii]
            fok = [int(field_of(pred.get(i)) == field_of(gold_answer(G[i]))) for i in ii]
            r = {"n": len(ii), "acc": sum(ok) / max(1, len(ii)), "field": sum(fok) / max(1, len(ii))}
            if conf is not None:
                r["ece"] = ece([float(conf.get(i) or 0.0) for i in ii], ok)
            out[(st, tier)] = r
    return out


def mcnemar(split, a, b, st="random", tier="agreed"):
    """Paired comparison of two prediction dicts on one tier: (only a right, only b right, z)."""
    G = gold(split); ids = [i for i in eval_ids(split) if i in G and G[i]["set"] == st and G[i]["tier"] == tier]
    oa = sum(1 for i in ids if a.get(i) == gold_answer(G[i]) and b.get(i) != gold_answer(G[i]))
    ob = sum(1 for i in ids if b.get(i) == gold_answer(G[i]) and a.get(i) != gold_answer(G[i]))
    return oa, ob, (oa - ob) / math.sqrt(oa + ob) if oa + ob else 0.0


# ---- the methods -----------------------------------------------------------------------------------------------

def student(split, tag="q8b_2m"):
    """A student's argmax over all 4,517 classes and its probability (from data/student_outputs)."""
    p, c = {}, {}
    for r in read_jsonl(os.path.join(DATA, "student_outputs", split, f"{tag}.jsonl.gz")):
        p[r["id"]], c[r["id"]] = r["top"][0][0], r["top"][0][1]
    return p, c


def old_model(split):
    """The previous OpenAlex topic model's primary topic as served when the gold works were fetched (None = no topic)."""
    p, c = {}, {}
    for w in read_jsonl(os.path.join(DATA, "gold", f"{split}_works.jsonl")):
        t = w.get("old_model_primary_topic") or {}
        p[w["id"]] = t.get("id") if t.get("id") in TOPIC else None
        c[w["id"]] = t.get("score") or 0.0
    return p, c


def shortlists(split):
    return {r["id"]: r for r in read_jsonl(os.path.join(DATA, "shortlist", f"{split}_jev_k255.jsonl.gz"))}


def retriever_top1(split):
    return {i: r["candidates"][0] for i, r in shortlists(split).items()}, None


def jev_alone(split):
    p, c = {}, {}
    for i, r in shortlists(split).items():
        k = max(range(len(r["p"])), key=lambda j: r["p"][j])
        p[i], c[i] = r["candidates"][k], r["p"][k]
    return p, c


def teacher_arm(split, arm):
    """A teacher arm's answer per work: the chosen topic, NOT_CLASSIFIABLE, or None (NONE_FIT, refused, error).
    Rows were appended as runs resumed; a work's first ok-or-refused row is its answer."""
    rows = {}
    for r in read_jsonl(os.path.join(DATA, "teacher_arms", split, f"{arm}.jsonl.gz")):
        if r["id"] not in rows or rows[r["id"]]["status"] not in ("ok", "refused"):
            rows[r["id"]] = r
    p, c = {}, {}
    for i, r in rows.items():
        a = r.get("answer") or {}
        t = a.get("topic_id") if r["status"] == "ok" else None
        p[i] = t if (t in TOPIC or t == NC_LABEL) else None
        c[i] = float(a.get("confidence") or 0.0)
    return p, c, rows


def fmt(x, d=3):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"
