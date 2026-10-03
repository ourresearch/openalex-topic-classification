"""Outside check on evidence no model produced: arXiv authors choose a primary category when they submit. For 2,000
random arXiv works with abstracts, does the topic's field match the category? The category-to-field map below was
written before any scoring; catch-all categories are excluded. Field level only (arXiv categories are coarse).
Usage: python3 eval/arxiv.py"""
import collections
from common import *

def F(*x): return set(str(v) for v in x)
EXACT = {"cs.CY": F(17, 33), "cs.SI": F(17, 33), "cs.DL": F(17, 33), "cs.GT": F(17, 20, 26, 18), "cs.CE": F(17, 22), "cs.SD": F(17, 22),
 "cs.SY": F(17, 22), "cs.RO": F(17, 22), "cs.NA": F(17, 26), "cs.IT": F(17, 26, 22), "cs.LO": F(17, 26), "cs.DM": F(17, 26), "cs.CC": F(17, 26),
 "cs.ET": F(17, 22), "cs.AR": F(17, 22), "cs.MS": F(17, 26), "cs.CG": F(17, 26), "cs.FL": F(17, 26), "cs.SC": F(17, 26), "cs.HC": F(17, 32), "cs.CL": F(17, 12),
 "math.MP": F(26, 31), "math.IT": F(26, 17, 22), "math.OC": F(26, 22, 18, 17), "math.ST": F(26, 18), "math.NA": F(26, 17), "math.HO": F(26, 12),
 "math.CO": F(26, 17), "math.LO": F(26, 17), "math.DS": F(26, 31), "math.AP": F(26, 31), "stat.ME": F(26, 18), "stat.ML": F(26, 18, 17), "stat.CO": F(26, 18, 17),
 "physics.chem-ph": F(31, 16), "physics.bio-ph": F(31, 13, 28), "physics.med-ph": F(31, 27), "physics.ao-ph": F(31, 19, 23), "physics.geo-ph": F(31, 19),
 "physics.soc-ph": F(31, 33, 17), "physics.data-an": F(31, 17, 26), "physics.comp-ph": F(31, 17, 26), "physics.flu-dyn": F(31, 22), "physics.app-ph": F(31, 22, 25),
 "physics.optics": F(31, 22), "physics.ins-det": F(31, 22), "physics.plasm-ph": F(31, 21), "physics.ed-ph": F(31, 33), "physics.hist-ph": F(31, 12), "physics.atm-clus": F(31, 16),
 "astro-ph.EP": F(31, 19), "cond-mat.soft": F(31, 25, 16), "cond-mat.stat-mech": F(31), "cond-mat.dis-nn": F(31, 17), "cond-mat.quant-gas": F(31),
 "q-bio.NC": F(28, 13), "q-bio.PE": F(11, 13, 23), "q-bio.QM": F(13, 17, 26), "q-bio.BM": F(13, 16), "q-bio.TO": F(13, 27), "q-bio.OT": F(13, 11, 27),
 "eess.IV": F(17, 22, 27), "eess.AS": F(17, 22), "eess.SP": F(22, 17), "eess.SY": F(22, 17)}
ARCHIVE = {"cs": F(17), "math": F(26), "math-ph": F(26, 31), "stat": F(26, 18, 17), "physics": F(31), "astro-ph": F(31), "cond-mat": F(31, 25), "quant-ph": F(31, 17),
 "hep-ph": F(31), "hep-th": F(31), "hep-ex": F(31), "hep-lat": F(31), "gr-qc": F(31), "nucl-th": F(31), "nucl-ex": F(31), "nlin": F(31, 26),
 "q-bio": F(13, 28, 11, 24, 27), "q-fin": F(20, 26, 18), "econ": F(20, 18), "eess": F(22, 17)}
EXCLUDE = {"stat.AP", "physics.gen-ph", "cs.GL", "math.GM", "q-bio.OT", "stat.OT"}
def allowed(cat):
    if not cat or cat in EXCLUDE: return None
    return EXACT.get(cat) or ARCHIVE.get(cat.split(".")[0])
FIELD = {t: TOPIC[t]["field"]["id"] for t in TOPIC}
W = list(read_jsonl(os.path.join(DATA, "arxiv", "arxiv_works.jsonl.gz")))
rows = [(w, allowed(w["arxiv_primary_category"])) for w in W]; rows = [(w, a) for w, a in rows if a]
def stu(w): t = (w.get("student_top10") or [[None]])[0][0]; return t
def tea(w):
    r = w.get("teacher") or {}; return (r.get("answer") or {}).get("topic_id") if r.get("status") == "ok" else None
def old(w): return w.get("old_model_primary_topic")
M = [("previous OpenAlex topic model", old), ("student", stu), ("teacher (Opus 5.5 xhigh on the student's top 10)", tea)]
print(f"{len(rows):,} of {len(W):,} arXiv works scorable\n\n| model | field matches the author's arXiv category |\n|---|---|")
for name, f in M: print(f"| {name} | {sum(FIELD.get(f(w)) in a for w, a in rows) / len(rows):.3f} |")
by = collections.defaultdict(list)
for w, a in rows: by[w["arxiv_primary_category"].split(".")[0]].append((w, a))
print("\n| archive | works | previous model | student | teacher |\n|---|---|---|---|---|")
for arch, rs in sorted(by.items(), key=lambda x: -len(x[1]))[:8]:
    print(f"| {arch} | {len(rs)} | " + " | ".join(f"{sum(FIELD.get(f(w)) in a for w, a in rs) / len(rs):.3f}" for _, f in M) + " |")
