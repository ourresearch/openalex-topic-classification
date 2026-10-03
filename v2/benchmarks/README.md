# Benchmarks

Every number in the main [README](../README.md), how we got it, and how to check it. Each table below is printed by a
script in [`eval/`](../eval/) from the files in [`data/`](../data/) (standard library, about a second each):

```
python3 eval/test_read.py          # the test table (--split dev for the development set)
python3 eval/panel.py              # the answer key: tiers and judge agreement
python3 eval/confidence.py         # scores, confident mistakes, not classifiable
python3 eval/arxiv.py              # the outside check against arXiv categories
python3 eval/teacher_arms.py       # choosing the teacher
python3 eval/students.py           # choosing the student
```

No titles or abstracts are included, because publishers own them; [`tools/fetch_works.py`](../tools/fetch_works.py)
rebuilds them from the OpenAlex API.

## The answer key

**Two frontier models label every test work blind; where they agree, that is the answer.** Claude Opus 5.5 (effort
xhigh) and GPT-6.1 Sol (effort high) each read the title, abstract, venue, year, type and language of a work and chose
its topic from the full taxonomy of 4,516 topics, written out in the prompt ([`gold/judge.py`](../gold/judge.py)).
Neither saw the other's answer or any candidate list. Where they differ, Claude Fable 5.1 votes the same way, and its
match wins ("tie-broken"); a three-way split is "no-consensus". Either judge may answer that the record is not
classifiable, and when both do, that is the answer. Opus 5.5 declines some biomedical papers; those were re-asked of
Claude Opus 5 and flagged.

| Set | Works | Agreed | Tie-broken | No-consensus | Unresolved | Judges agree on the topic | on the subfield | on the field |
|---|---|---|---|---|---|---|---|---|
| Test, random | 2,000 | 1,503 | 385 | 93 | 19 | 75.2% | 80.1% | 86.8% |
| Test, highly cited | 200 | 168 | 26 | 3 | 3 | 84.4% | 85.4% | 88.4% |
| Development, random | 999 | 766 | 185 | 39 | 9 | 76.8% | 80.9% | 87.6% |
| Development, highly cited | 200 | 172 | 22 | 1 | 5 | 86.0% | 91.0% | 94.5% |

**We report the agreed tier.** It is a large, clean answer key: two models from different labs reached the same
topic out of 4,516 independently. The tie-broken tier is harder and noisier, and its tie-breaker comes from the same
lab as the teacher, which may favour the teacher; its numbers are in every table but we don't lead with them.

**The samples.** Test: 2,000 works drawn uniformly from every work with a title (`xxhash64(id, 1485) % 200000 = 0`,
the first 2,000 by hash) and 200 works with at least 111 citations, the top 1% (`xxhash64(id, 14851) % 20000 = 0`),
drawn on 1 October 2026 before any labelling, with no overlap with the development set or any training work.
Development: 1,198 works drawn the same way in September 2026 for an earlier benchmark; their tie-breaking vote is an
earlier Fable 5.1 label made in two passes with different taxonomy orders and adjudicated. The development set was
used for every choice; the test set was scored once, after the model was final. Ids, every judge's raw answer and the
gold are in [`data/gold/`](../data/gold/); `python3 gold/build_gold.py --split test` rebuilds the gold from the judges'
rows.

## The test read

Primary-topic accuracy. A missing answer, a refusal or "none of these fit" counts as wrong. "Not classifiable" is right
when the judges agreed the record is not classifiable.

| Method | Random, agreed (n = 1,503) | Field | Calibration error | Highly cited, agreed (n = 168) | Random, tie-broken (n = 385) |
|---|---|---|---|---|---|
| Previous OpenAlex topic model (v1) | 24.8% | 44.0% | 0.271 | 70.2% | 14.8% |
| Retriever alone (top 1) | 48.7% | 69.6% | | 63.7% | 19.2% |
| Jev alone, 255 candidates | 70.5% | 81.8% | 0.033 | 79.8% | 28.6% |
| **New model (v2)** | **81.7%** | **89.6%** | **0.047** | **89.3%** | **43.9%** |
| Same model, first million labels only | 81.6% | 90.3% | 0.038 | 89.3% | 44.4% |
| Teacher: Claude Opus 5.5 (xhigh) choosing from Jev's shortlist | 86.7% | 92.1% | 0.214 | 93.5% | 53.2% |

Calibration error is the expected calibration error of the top answer's score (15 equal bins). The previous model's
topics are the ones the API served for these works on 1 October 2026. With 1,503 works, an accuracy has a 95%
interval of about ±2 points.

**A lenient measure.** The judges also named up to two other topics that fit each work. Counting the new model right
when its primary topic is any topic the agreeing judges accepted, it is right on 91.8% of random works with a topic,
and the previous model on 37.3%.

**By kind of work** (random works, agreed tier):

| Works | n | Previous model | New model | Teacher |
|---|---|---|---|---|
| with an abstract | 896 | 28.1% | 86.6% | 90.2% |
| without an abstract | 607 | 19.9% | 74.5% | 81.5% |
| in English | 1,060 | 30.7% | 84.9% | 89.4% |
| in other languages | 376 | 9.3% | 71.8% | 79.3% |
| articles | 764 | 33.4% | 80.6% | 85.7% |
| other types | 739 | 16.0% | 82.8% | 87.7% |

## Scores and confident mistakes

**The new model's score is a probability you can use.** Random works, agreed tier:

| Score of the primary topic | Share of works | Right |
|---|---|---|
| 0.9 to 1 | 64% | 94.5% |
| 0.7 to 0.9 | 16% | 72.7% |
| 0.5 to 0.7 | 10% | 60.0% |
| under 0.5 | 10% | 35.9% |

**The previous model was often wrong when it was sure.** On the same works:

| Previous model's score | Works | Previous model right | New model right |
|---|---|---|---|
| at least 0.5 | 780 | 43.1% | 83.7% |
| at least 0.9 | 578 | 49.1% | 84.1% |
| at least 0.95 | 499 | 52.9% | 85.2% |

So keeping the previous model's confident answers would not help: the new model replaces every assignment.

**Not classifiable.** The new model says a record is not classifiable for 109 of the 1,503 random works (7%); the
judges agree on 77 of them. The previous model gave no topic at all to 308 of the 1,503 and a topic to all the rest,
including records with nothing to classify.

## Outside check: arXiv categories

**Evidence no model produced.** When authors submit to arXiv they choose a primary category (cs.CL, math.AP,
astro-ph.GA and so on). We drew 2,000 random arXiv papers with abstracts from OpenAlex, fetched each one's category
from the arXiv API, and wrote a map from categories to the OpenAlex fields they allow before scoring anything
([`eval/arxiv.py`](../eval/arxiv.py); catch-all categories such as stat.AP are left out, leaving 1,984). A topic counts
as right when its field is allowed. This is a field-level check on English, mostly STEM papers.

| Model | Field matches the authors' category |
|---|---|
| Previous OpenAlex topic model | 74.8% |
| **New model** | **82.7%** |
| Teacher (Claude Opus 5.5, choosing from the new model's top 10) | 82.1% |

| Archive | Papers | Previous model | New model | Teacher |
|---|---|---|---|---|
| cs | 856 | 71.6% | 79.9% | 79.0% |
| math | 459 | 77.6% | 84.5% | 83.7% |
| physics | 133 | 63.2% | 71.4% | 71.4% |
| cond-mat | 84 | 78.6% | 83.3% | 79.8% |
| astro-ph | 74 | 85.1% | 97.3% | 97.3% |
| eess | 72 | 87.5% | 88.9% | 87.5% |
| quant-ph | 69 | 88.4% | 94.2% | 95.7% |
| stat | 47 | 68.1% | 80.9% | 83.0% |

## Choosing the teacher

**On the development set, Claude Opus 5.5 at effort xhigh, choosing from Jev's p ≥ 0.02 shortlist, scored highest;
the test set confirmed it.** Each arm is a model, an effort setting and a shortlist rule: p01 to p05 keep every
candidate Jev gives at least that probability (at least 2, at most 30); top60 is the retriever's top 60 with no Jev;
q17top10 is the top 10 of a small student. Refusals count as wrong. `python3 eval/teacher_arms.py`; the answers are in
[`data/teacher_arms/`](../data/teacher_arms/).

| Arm (development set) | Random, agreed | Field | Highly cited, agreed | Random, tie-broken | Refused | Mean options |
|---|---|---|---|---|---|---|
| Jev alone, 255 candidates | 69.2% | 82.5% | 76.2% | 30.8% | 0 | 255 |
| **Opus 5.5 xhigh, p02** | **86.6%** | **92.4%** | **94.2%** | **50.8%** | 1.0% | 6.7 |
| Opus 5.5 xhigh, q17top10 | 87.1% | 92.6% | 93.6% | 49.7% | 1.0% | 10.0 |
| Opus 5.5 high, p02 | 85.4% | 91.4% | 95.3% | 48.6% | 0.8% | 6.7 |
| Opus 5.5 medium, p01 | 86.4% | 91.8% | 93.6% | 49.2% | 0.9% | 8.6 |
| Opus 5.5 medium, p02 | 86.0% | 92.0% | 93.6% | 48.1% | 0.9% | 6.7 |
| Opus 5.5 medium, p03 | 85.8% | 91.5% | 94.8% | 48.1% | 0.7% | 4.2 |
| Opus 5.5 medium, p05 | 84.3% | 91.3% | 94.2% | 46.5% | 0.8% | 3.0 |
| Opus 5.5 medium, top60 | 83.0% | 89.6% | 92.4% | 44.9% | 1.0% | 60.0 |
| Opus 5 xhigh, p02 | 85.5% | 91.8% | 94.2% | 49.2% | 0.3% | 6.7 |
| GPT-6 Astra high, p02 | 82.9% | 89.7% | 93.6% | 38.4% | 0 | 6.7 |
| GPT-6 Astra xhigh, p02 | 82.8% | 89.3% | 91.9% | 36.2% | 0 | 6.7 |
| GPT-6.1 Sol high, p02 | 82.0% | 88.9% | 90.7% | 36.2% | 0 | 6.7 |
| GPT-6.1 Sol medium, p02 | 81.2% | 88.3% | 91.3% | 35.1% | 0 | 6.7 |
| GPT-6.1 Sol medium, p01 | 82.0% | 88.6% | 90.7% | 35.1% | 0 | 8.6 |
| GPT-6.1 Sol medium, p03 | 81.5% | 87.9% | 87.8% | 34.6% | 0 | 4.2 |
| GPT-6.1 Sol medium, p05 | 79.5% | 86.8% | 89.0% | 35.7% | 0 | 3.0 |
| GPT-6.1 Sol medium, top60 | 80.5% | 87.3% | 91.3% | 34.6% | 0 | 60.0 |

| Arm (test set) | Random, agreed | Field | Highly cited, agreed | Random, tie-broken | Refused |
|---|---|---|---|---|---|
| Jev alone, 255 candidates | 70.5% | 81.8% | 79.8% | 28.6% | 0 |
| **Opus 5.5 xhigh, p02** | **86.7%** | **92.1%** | **93.5%** | **53.2%** | 0.9% |
| Opus 5.5 medium, p02 | 85.0% | 91.2% | 92.9% | 52.5% | 1.0% |
| GPT-6.1 Sol medium, p02 | 84.0% | 89.6% | 91.1% | 40.0% | 0 |

Paired comparisons on random works, agreed tier (works only one arm got right, and the z of a McNemar test): on the
test set, xhigh against medium 49 to 23 (z = 3.1). On the development set, a shortlist beats no shortlist (p02 against
top60 at medium, 56 to 33, z = 2.4) and p02 beats p05 (27 to 14, z = 2.0); p01, p02 and p03 are within noise of each
other, so p02 is the shortest list that keeps the right topic (97% of the time on test, about 7 options on average). The
small student's top 10 ties Jev's shortlist (34 to 30, z = 0.5), so the second million of labels used it and needed
no Jev.

**Shortlist recall** (random, agreed, test): the retriever's top 255 holds the right topic 98.5% of the time, Jev's
p ≥ 0.02 shortlist 96.6%, the new model's own top 3, 10 and 20 92.1%, 96.9% and 98.5%.

## Choosing the student

**Bigger backbones and more labels both help, mostly on rare topics; no change to the training recipe did.** Teacher
fidelity is the share of 20,000 held-out training works (work id ending in 7, never trained on) where the student
gives the teacher's answer; it separates differences of a point that the development set cannot. The band columns
split those works by how many training labels the teacher's answer had in the first million.
`python3 eval/students.py`

| Student | Fidelity | < 50 labels | 50 to 200 | 200 to 1K | > 1K | Development, random, agreed | Field | Calibration error | Highly cited |
|---|---|---|---|---|---|---|---|---|---|
| multilingual-e5-base classifier, 1M labels | 62.3% | 23.3% | 50.0% | 65.1% | 82.4% | 69.8% | 83.0% | 0.078 | 73.3% |
| Qwen3-1.7B, 1M labels | 72.8% | 42.9% | 65.4% | 74.5% | 85.9% | 77.3% | 88.0% | 0.050 | 83.7% |
| Qwen3-4B, 1M labels | 75.6% | 44.9% | 68.9% | 77.6% | 86.3% | 79.5% | 88.9% | 0.048 | 82.6% |
| Qwen3-8B, 1M labels | 76.7% | 44.9% | 69.8% | 79.1% | 86.3% | 80.4% | 89.2% | 0.054 | 88.4% |
| Qwen3-1.7B, 2M labels | 73.1% | 50.5% | 65.9% | 74.3% | 86.2% | 77.7% | 87.6% | 0.044 | 83.7% |
| **Qwen3-8B, 2M labels (released)** | **76.8%** | **55.3%** | **70.7%** | **78.3%** | **87.1%** | **79.6%** | **89.2%** | **0.068** | **88.4%** |

Held-out works per band: 515, 5,364, 11,384 and 2,737. The released model was chosen before the test read: it equals
the 8B on the first million on every measure with power (development set: 33 works only it got right, 39 only the
other, z = −0.7) and is 10 points better on the rarest topics.

One change at a time to the Qwen3-1.7B recipe (1M labels; baseline fidelity 72.8%):

| Change | Fidelity | Development, random, agreed | Calibration error | Verdict |
|---|---|---|---|---|
| soft labels (0.2 on the teacher's other topics, 0.05 smoothing) | 72.8% | 78.1% | 0.080 | no gain |
| mean pooling | 72.9% | 76.4% | 0.052 | no gain |
| 512 tokens | 72.9% | 77.4% | 0.049 | no gain |
| 2 epochs | 72.4% | 77.7% | 0.106 | no gain, overconfident |
| head initialised from topic texts | 73.0% | 78.1% | 0.036 | no gain |
| subfield and field auxiliary loss | 71.8% | 76.2% | 0.069 | worse |
| learning rate 4e-5 | 71.8% | 77.3% | 0.042 | worse |
| half the labels (500K) | 70.4% | 72.7% | 0.068 | labels matter |

So the released recipe is the plain one: a classification head on the last token, the teacher's answer as a hard
label, one epoch, 384 tokens.

## Rare topics

The first million labels cover 4,502 of the 4,516 topics; the median topic has 147 labels and 301 have fewer than
20. The second million was aimed at the thin ones (every topic was topped up towards 431 labels where the pool had
candidates), and it lifted teacher fidelity on the rarest band from 44.9% to 55.3% with no loss elsewhere. 14 topics
have no label in either million, so the model never assigns them. All are vague catch-alls:

T12705 Educational Reforms and Innovations · T14046 Bach Studies and Logistics Development · T13431 Education Methods
and Integration · T14016 International Relations and Autism · T14264 Law, Ethics, and AI Impact · T14413 Advanced
Technologies in Various Fields · T13550 Diverse Applied Research Studies · T14073 Diverse Global Research Studies ·
T14023 Interdisciplinary Studies and Sociocultural Dynamics · T14122 Impact of Education Environments · T13680
Education, Psychology, and Complexity Research · T14202 Sustainability, Governance, and Employment Studies · T13612
Advanced Scientific and Engineering Studies · T14087 Economic, Educational, Environmental and Organizational
Development

Label counts per topic are in [`data/teacher_labels/label_counts.json`](../data/teacher_labels/label_counts.json).

## The production path

OpenAlex tags the corpus with vLLM and FP8 weights ([`student/score_vllm.py`](../student/score_vllm.py)), which is
faster than 16-bit; the numbers above are from the 16-bit model. The FP8 path gives the same results to within noise:

| Path (released weights) | Random, agreed | Field | Calibration error | Highly cited, agreed | Teacher fidelity (20,000 held-out) | Works a second, one H100 |
|---|---|---|---|---|---|---|
| Hugging Face, bf16 (`student/infer.py`) | 81.8% | 89.6% | 0.047 | 89.3% | 76.9% | |
| vLLM, FP8 (`student/score_vllm.py`) | 81.6% | 89.6% | 0.041 | 89.3% | 76.9% | about 300 |

Both rows were run on 3 October 2026 from the released weights by the repo's own scripts, on test texts fetched from
the public API that day; the fidelity column on the training texts. FP8 gives the same top topic as bf16 on 98.5% of
the test works.
