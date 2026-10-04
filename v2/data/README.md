# Data

Everything the benchmarks are computed from. CC0. No titles or abstracts: `tools/fetch_works.py` rebuilds them from
the OpenAlex API. Topic ids are OpenAlex topic ids (`T10017` is https://openalex.org/T10017).

| Path | What it is |
|---|---|
| `topics/topics.json` | The 4,516 topics with names, descriptions, keywords and hierarchy, in the model's class order (class i = topic i; class 4516 = NOT_CLASSIFIABLE). Unchanged from OpenAlex's topics. |
| `gold/{dev,test}_works.jsonl` | The gold works: id, set (`random` or `cited`), year, type, language, whether the record had an abstract and a venue when it was judged, and the topics version 1 served for it then (`old_model_*`). |
| `gold/judges/{split}_{judge}.jsonl.gz` | Every judge's raw answer: `opus` = Claude Opus 5.5 (xhigh), `sol` = GPT-6.1 Sol (high), `fable` = Claude Fable 5.1 (xhigh), `opus5` = Claude Opus 5 where Opus 5.5 declined. Each answer has the topic, up to two secondary topics, a level, a confidence and a one-sentence reason. |
| `gold/panel_gold_{split}.jsonl` | The answer key built from them by `gold/build_gold.py`: tier, gold topic, acceptable topics, every vote. |
| `shortlist/{split}_jev_k255.jsonl.gz` | For each gold work, the retriever's top 255 topics in rank order and Jev's probability for each. |
| `teacher_arms/{split}/{arm}.jsonl.gz` | Every teacher arm's answers on the gold works, with the shortlist it saw (`cands`). |
| `student_outputs/{split}/{student}.jsonl.gz` | Each student's top 20 classes and probabilities on the gold works (`q8b_2m` is the released model). |
| `student_outputs/heldout/{student}.jsonl.gz` | Each student's answer and probability on 20,000 held-out training works, with the teacher's answer. |
| `arxiv/arxiv_works.jsonl.gz` | The arXiv outside check: id, arXiv URL, the authors' primary category, version 1's topic, the new model's top 10, the teacher's answer. |
| `corpus/topic_counts.csv` | For every topic: works whose primary topic it is under the previous model and the new one, and works with it in the new model's top three (all 473,917,182 scored works, October 2026). |
| `teacher_labels/label_counts.json` | Teacher labels per topic in each million (all, and training rows only). |
| `teacher_labels/second_million_selection.json` | How the second million was picked: the target per topic and the candidates available. |
| `teacher_labels/first_million_pool_strata.json` | The strata of the pool the first million was drawn from. |

**The 2M teacher labels** are a release asset, `teacher_labels_2m.jsonl.gz` (`python3 tools/download.py labels`). One
row per work: `million` (1 or 2), `sample` (`uniform` or `rarity` in the first million, `tail` in the second),
`stratum` and `is_xpac` (first million), `target_topic_id` (second million), `shortlist` (`jev_p02` or
`student_top10`), `candidates` (exactly what the teacher was shown, in order), `topic_id` (a topic, `NOT_CLASSIFIABLE`
or `NONE_FIT`), `secondary_topic_ids`, `confidence` (the teacher's own, poorly calibrated), `model` (which model made the
label) and `heldout` (work id ending in 7: never trained on).

**How the first million was drawn.** A pool of 1.22 million works from all of OpenAlex (title of at least 10
characters), stratified by script (CJK, Cyrillic, other non-Latin, Latin), abstract presence, and year, language or
type, with quotas that over-represent what is hard to classify (`first_million_pool_strata.json`). 750,000 were drawn
uniformly from the shuffled pool; 250,000 more were picked for rare topics (for every topic, rarest first, up to 250
works whose retriever top 30 held it). The second million is described in `teacher/select_tail.py`.
