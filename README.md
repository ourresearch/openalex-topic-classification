# openalex-topic-classification

How [OpenAlex](https://openalex.org) assigns topics to works. Every work gets up to three of 4,516 topics, each in a
subfield, a field and a domain. Each model version has its own folder with everything needed to replicate it. To learn
more about topics in OpenAlex, see the [help pages](https://help.openalex.org/data/topics/).

### Model versions
* [v2](v2/) (current, October 2026): a Qwen3-8B classifier trained on 2 million topic choices by Claude Opus 5.5;
  right on 82% of random works where version 1 was right on 25%. Code, open weights, teacher labels, test sets and
  every benchmark are in [v2/](v2/).
* [v1](v1/) (2024, superseded): multilingual BERT fine-tuned on data labelled by CWTS.

### Topics
The topics themselves are the same in every version: 4,516 topics with names, keywords, descriptions and a
hierarchy, [listed here](v2/data/topics/topics.json) and served by the [API](https://api.openalex.org/topics). They
were built with CWTS (Leiden University); see [An open approach for classifying research publications](https://www.leidenmadtrics.nl/articles/an-open-approach-for-classifying-research-publications).
