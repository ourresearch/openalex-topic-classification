"""Draw the README's two charts as static SVGs from the files in data/ (standard library).

    python3 docs/charts/make_charts.py

The numbers come from the same functions as eval/test_read.py and eval/confidence.py, so the charts and the tables
always agree.
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
OUT = HERE.parent / "img"
sys.path.insert(0, str(ROOT / "eval"))
from common import TOPIC, eval_ids, gold, gold_answer, old_model, score, student, read_jsonl, DATA   # noqa: E402

# One version per chart that reads on GitHub's light (#ffffff) and dark (#0d1117) pages alike (a <picture> with a
# dark variant follows the operating system, not the GitHub theme). Text #707a84 is 4.4:1 on both; the two bar
# colors clear 3:1 on both and pass a colour-blind separation check.
THEME = dict(ink="#707a84", grid="#8b949e", old="#199e70", new="#2a78d6")
FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif'


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, w, h, title):
        self.w, self.h, self.parts, self.title = w, h, [], title

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, fill, size=14, anchor="start", weight=400):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
                 f'text-anchor="{anchor}" dominant-baseline="middle">{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke, opacity=0.35):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
                 f'stroke-width="1" stroke-opacity="{opacity}"/>')

    def save(self, path):
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
            f'role="img" font-family=\'{FONT}\' style="font-variant-numeric: tabular-nums">\n'
            f'<title>{esc(self.title)}</title>\n' + "\n".join(self.parts) + "\n</svg>\n")


def hbar(x0, y, w, h, r=4):
    """Bar from the baseline x0, square at the baseline, rounded at the data end."""
    if w < 0.5:
        return ""
    r = min(r, w / 2, h / 2)
    return (f"M{x0:.1f},{y:.1f} H{x0 + w - r:.1f} Q{x0 + w:.1f},{y:.1f} {x0 + w:.1f},{y + r:.1f} "
            f"V{y + h - r:.1f} Q{x0 + w:.1f},{y + h:.1f} {x0 + w - r:.1f},{y + h:.1f} H{x0:.1f} Z")


def legend(s, x, y, items, t):
    for label, color in items:
        s.add(f'<rect x="{x}" y="{y - 6}" width="12" height="12" rx="2" fill="{color}"/>')
        s.text(x + 18, y, label, t["ink"], 14)
        x += 18 + len(label) * 7.6 + 26


def grouped(t, title, rows, series, fmt, sub=None, L=250):
    """Grouped horizontal bars on a visible 0-100% scale, one row per group."""
    W, R, bh, gap, pad = 720, 56, 14, 4, 22
    n = len(series)
    top = 62
    rowH = n * bh + (n - 1) * gap + pad
    H = top + len(rows) * rowH + 4
    s = Svg(W, H, title)
    x = lambda v: L + v / 100 * (W - L - R)
    legend(s, L, 14, series, t)
    for tick in (0, 25, 50, 75, 100):
        s.line(x(tick), top - 12, x(tick), H - 4, t["grid"], opacity=0.8 if tick == 0 else 0.35)
        s.text(x(tick), top - 22, f"{tick}%", t["ink"], 12, "middle")
    for i, (label, vals) in enumerate(rows):
        y0 = top + i * rowH + pad / 2
        cy = y0 + (n * bh + (n - 1) * gap) / 2
        s.text(0, cy, label, t["ink"], 15, weight=600)
        for j, v in enumerate(vals):
            y = y0 + j * (bh + gap)
            s.add(f'<path d="{hbar(x(0), y, x(v) - x(0), bh)}" fill="{series[j][1]}"/>')
            s.text(x(v) + 6, y + bh / 2, fmt(v), t["ink"], 13, weight=600 if j == n - 1 else 400)
    return s


def arxiv():
    import os, importlib.util
    spec = importlib.util.spec_from_file_location("arxiv_eval", ROOT / "eval" / "arxiv.py")
    src = (ROOT / "eval" / "arxiv.py").read_text()
    ns = {}
    exec(src.split("W = list(")[0], ns)   # the category map and allowed(), without printing
    FIELD = {t: TOPIC[t]["field"]["id"] for t in TOPIC}
    rows = [(w, ns["allowed"](w["arxiv_primary_category"])) for w in read_jsonl(os.path.join(DATA, "arxiv", "arxiv_works.jsonl.gz"))]
    rows = [(w, a) for w, a in rows if a]
    old = sum(FIELD.get(w.get("old_model_primary_topic")) in a for w, a in rows) / len(rows)
    new = sum(FIELD.get((w.get("student_top10") or [[None]])[0][0]) in a for w, a in rows) / len(rows)
    return old, new


def main():
    t = THEME
    OUT.mkdir(exist_ok=True)
    series = [("Previous model", t["old"]), ("New model", t["new"])]
    pct = lambda v: f"{v:.0f}%"
    so, sn = score("test", *old_model("test")), score("test", *student("test"))
    ao, an = arxiv()
    rows = [("Topic, random works", [100 * so[("random", "agreed")]["acc"], 100 * sn[("random", "agreed")]["acc"]]),
            ("Field, random works", [100 * so[("random", "agreed")]["field"], 100 * sn[("random", "agreed")]["field"]]),
            ("Topic, highly cited works", [100 * so[("cited", "agreed")]["acc"], 100 * sn[("cited", "agreed")]["acc"]]),
            ("Field vs arXiv authors' category", [100 * ao, 100 * an])]
    grouped(t, "How often the primary topic is right: previous OpenAlex topic model and the new one", rows, series, pct).save(OUT / "accuracy.svg")
    G = gold("test"); ids = [i for i in eval_ids("test") if i in G and G[i]["set"] == "random" and G[i]["tier"] == "agreed"]
    (op, oc), (sp, sc) = old_model("test"), student("test")
    def acc(p, c, lo, hi):
        ii = [i for i in ids if p.get(i) and lo <= (c.get(i) or 0) < hi]
        return 100 * sum(p[i] == gold_answer(G[i]) for i in ii) / max(1, len(ii))
    bands = [("Score 0.9 to 1", 0.9, 1.01), ("Score 0.7 to 0.9", 0.7, 0.9), ("Score 0.5 to 0.7", 0.5, 0.7), ("Score under 0.5", 0.0, 0.5)]
    rows = [(label, [acc(op, oc, lo, hi), acc(sp, sc, lo, hi)]) for label, lo, hi in bands]
    grouped(t, "How often the primary topic is right on random works, by the score the model gave it", rows, series, pct).save(OUT / "confidence.svg")
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.svg"))))


if __name__ == "__main__":
    main()
