# What changed: topics before and after version 2

On 6 October 2026 OpenAlex replaced every work's topics with the [version 2 model](../). **The topics themselves did
not change**: same 4,516 topics, names, IDs, subfields, fields and domains. What changed is which works are in each.
These files show where works moved, so you can check your own reports.

Counts are by **primary topic** (each work counts once), over every work the new model scored (475,958,097 works with
a title or abstract, 5 October 2026). "Old" is version 1's primary topic on 5 October; "new" is version 2's.

| File | What it is |
|---|---|
| [`topics.csv`](topics.csv) | Every topic: works before and after, change, % change, how many of its old works are still there (count and share), works with it among their top 3 topics, before and after, and how often an independent judge agrees with its works, before and after (see below). |
| [`subfields.csv`](subfields.csv), [`fields.csv`](fields.csv), [`domains.csv`](domains.csv) | The same, rolled up by each topic's place in the hierarchy (judge columns for subfields and fields). |
| [`topic_transitions.csv`](topic_transitions.csv) | For every old topic, the 10 new topics that received most of its works, with counts and shares, and the rest as "other". `(no topic)` means the new model found nothing to classify. |
| [`subfield_transitions.csv`](subfield_transitions.csv), [`field_transitions.csv`](field_transitions.csv), [`domain_transitions.csv`](domain_transitions.csv) | The same at each level. |
| [`countries_by_field.csv`](countries_by_field.csv) | Every country's publications by field, before and after: works and share of its works with a topic. |
| [`institutions/`](institutions/) | Before-and-after profiles for the 109 institutions on our [supporters page](https://openalex.org/institutional-supporters): fields, subfields and top topics. |

Country and institution files count publications only (articles, reviews, books, chapters, preprints, letters,
editorials, reports, dissertations, reference entries) and leave out datasets and catalogue records, since those are
what research reports count. Institutions include their child institutions.

**The headline.** Of works with a primary topic under both models, 29% keep the same topic, 37% the same subfield, 53%
the same field and 73% the same domain. The two biggest old topics were catch-alls and empty out: "Military Technology
and Strategies" (20.6M works, mostly non-English papers the old model couldn't read, now 170K) and "Diverse Scientific
and Economic Studies" (4.8M, now 80). 85 catch-all topics now have no works; they stay in the list, unchanged. Topics
that grow most are mostly ones that now also hold datasets and catalogue records (fusion-device shot records,
specimen records).

**Judge columns.** For every topic we sampled up to 6 works whose primary topic it was under the old model and up to
6 under the new one (53,621 works in all, none used to train the model). GPT-6.1 Sol read each work's title, abstract
and venue blind and chose up to 3 topics from all 4,516. `judged_old` / `judged_new` count the works it judged;
`judge_agrees_old` / `judge_agrees_new` are the share where the topic was among its picks; `judge_agrees_change_pts`
is the difference in points. With 6 works per side, one topic's numbers are noisy: read them as a flag, and use the
subfield and field rows (the share where the judge's main topic is in the same subfield or field, pooled over the
level's topics and weighted by works) for anything firmer. Over all topics, weighted by works: the judge agrees with
31% of old assignments and 77% of new ones; 82% of topics improve and 6% get worse. Like every test built on reading
the text, this measures how well a topic matches what a work says, not where it sits in the citation network.

**Old topics.** Every work's version 1 topics and scores are attached to the
[v2.0.0 release](https://github.com/ourresearch/openalex-topic-classification/releases/tag/v2.0.0) as 12 gzipped CSV
files, `topics_v1_frozen_part00.csv.gz` to `topics_v1_frozen_part11.csv.gz` (about 740 MB each, 8.9 GB in all;
checksums in `SHA256SUMS`). Columns: `work_id, topic_1_id, topic_1_score, topic_2_id, topic_2_score, topic_3_id,
topic_3_score`; `topic_1_id` is the old primary topic. 526,108,668 works: the latest version 1 assignment for each work
as of 5 October 2026. Use them to reproduce a report made with the old topics. The
[text aboutness endpoint](https://help.openalex.org/api/tag-aboutness/) also keeps the old classifier at
`/text/topics?version=1` until 13 January 2027. The old classifier itself stays public: its code is in
[v1/](../../v1/) and its weights and training data are on [Zenodo](https://zenodo.org/records/10568402), so you can
keep running it.

Built from OpenAlex's pipeline tables `work_topics` (version 1) and `work_topics_v2` on 5 October 2026.
