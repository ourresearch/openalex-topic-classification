"""The headline table: every method on the fresh test set (2,000 random + 200 highly cited works, read once, after
the student was chosen on the development set). Standard library, about a second.
Usage: python3 eval/test_read.py [--split dev]"""
import sys
from common import *

split = "dev" if "--split" in sys.argv and sys.argv[sys.argv.index("--split") + 1] == "dev" else "test"
rows = [("previous OpenAlex topic model", *old_model(split)),
        ("retriever top 1", *retriever_top1(split)),
        ("Jev alone, 255 candidates", *jev_alone(split)),
        ("student (Qwen3-8B, 2M labels)", *student(split, "q8b_2m")),
        ("student (Qwen3-8B, first 1M labels only)", *student(split, "q8b_e1")),
        ("teacher (Opus 5.5 xhigh, Jev shortlist)", *teacher_arm(split, "opus_xhigh_p02")[:2])]
print(f"## {split}: primary-topic accuracy against the panel gold (refusals and missing answers count as wrong)\n")
print("| method | random, agreed | field | calibration error | cited, agreed | random, tie-broken |")
print("|---|---|---|---|---|---|")
res = {}
for name, p, c in rows:
    s = score(split, p, c); res[name] = s
    ra, ca, rt = s[("random", "agreed")], s[("cited", "agreed")], s[("random", "tie-broken")]
    print(f"| {name} | {fmt(ra['acc'])} | {fmt(ra['field'])} | {fmt(ra.get('ece')) if c is not None else ''} | {fmt(ca['acc'])} | {fmt(rt['acc'])} |")
n = res[rows[0][0]]
print(f"\nn: random agreed {n[('random', 'agreed')]['n']:,}, cited agreed {n[('cited', 'agreed')]['n']:,}, random tie-broken {n[('random', 'tie-broken')]['n']:,}")
