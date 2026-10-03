# Changelog

The topic model uses [semantic versioning](https://semver.org). A **major** version replaces the approach. A **minor**
version retrains the model (new labels, a new backbone or recipe) and re-tags the corpus. A **patch** fixes code or
documentation without changing any assignment. Every release reports its test read here. The topics themselves
(names, keywords, hierarchy, IDs) are not versioned here; no version of this model changes them.

## 2.0.0 (October 2026)

A new model replaces version 1 on every work: a Qwen3-8B classifier trained on 2 million topic choices made by Claude
Opus 5.5, reading each work's title, abstract and venue. First public release of its code, weights, teacher labels,
answer key and every benchmark.

Test read (2,000 random and 200 highly cited works, scored once, against two frontier models' agreed answers;
`python3 eval/test_read.py`): primary topic right on 81.7% of random works (version 1: 24.8%), 89.6% at the field
level (44.0%), 89.3% of highly cited works (70.2%); calibration error 0.047 (0.271). Field matches the authors' arXiv
category on 82.7% of 1,984 arXiv papers (74.8%).

What changes for users: every work's `topics` and `primary_topic` (and so its subfield, field and domain); `score` is
now the model's probability; works the model calls not classifiable get no topic; anything computed from topics
moves, including counts per topic and field-normalized citation impact.

## 1.0.0 (2024)

The first OpenAlex topic model: multilingual BERT fine-tuned on title and abstract, with citation and venue features
in the original design. Code and notebooks in [v1/](../v1/).
