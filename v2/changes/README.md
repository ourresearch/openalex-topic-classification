# What changed: topics before and after version 2

On 6 October 2026 OpenAlex replaced every work's topics with the [version 2 model](../). **The topics themselves did
not change**: same 4,516 topics, names, IDs, subfields, fields and domains. What changed is which works are in each.
These files show where works moved, so you can check your own reports.

Counts are by **primary topic** (each work counts once), over every work the new model scored (475,958,097 works with
a title or abstract, 5 October 2026). "Old" is version 1's primary topic on 5 October; "new" is version 2's.

| File | What it is |
|---|---|
| [`topics.csv`](topics.csv) | Every topic: works before and after, change, % change, how many of its old works are still there (count and share), and works with it among their top 3 topics, before and after. |
| [`subfields.csv`](subfields.csv), [`fields.csv`](fields.csv), [`domains.csv`](domains.csv) | The same, rolled up by each topic's place in the hierarchy. |
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

**Old topics.** Every work's version 1 topics and scores, frozen on 5 October 2026, are published as release assets.
[TK: link once Jason approves.]

Rebuild these files: [TK script path], from `work_topics` (version 1) and `work_topics_v2` in OpenAlex's pipeline.
