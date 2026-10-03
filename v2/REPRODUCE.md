# Reproduce

Written for a person or an AI agent starting from a fresh clone. Every number in the README and the benchmarks comes
from files in this folder: the judges' answers, the teacher's answers and the model's outputs are all saved, so no path
below needs an API key, a GPU or any private resource until you choose to call a model yourself. The repo holds no
titles or abstracts; [`tools/fetch_works.py`](tools/fetch_works.py) rebuilds them from the public OpenAlex API when a
step needs text.

The scoring scripts need only Python 3.9 or later. Steps that run models list their extra packages.

```bash
git clone https://github.com/ourresearch/openalex-topic-classification
cd openalex-topic-classification/v2
```

## 1. Every benchmark, from saved answers (seconds, a laptop)

```bash
python3 eval/test_read.py          # the README's test table
python3 eval/panel.py              # the answer key: tiers and how often the judges agree
python3 eval/confidence.py         # the score bands, the previous model's confident mistakes, not classifiable
python3 eval/arxiv.py              # the outside check against arXiv categories
python3 eval/teacher_arms.py       # choosing the teacher (development set, then test)
python3 eval/students.py           # choosing the student (teacher fidelity, development set, ablations)
python3 docs/charts/make_charts.py # redraws docs/img/*.svg
python3 gold/build_gold.py --split test --out /tmp/panel_gold_test.jsonl && cmp /tmp/panel_gold_test.jsonl data/gold/panel_gold_test.jsonl
```

`eval/test_read.py` should print:

```
| previous OpenAlex topic model | 0.248 | 0.440 | 0.271 | 0.702 | 0.148 |
| retriever top 1 | 0.487 | 0.696 |  | 0.637 | 0.192 |
| Jev alone, 255 candidates | 0.705 | 0.818 | 0.033 | 0.798 | 0.286 |
| student (Qwen3-8B, 2M labels) | 0.817 | 0.896 | 0.047 | 0.893 | 0.439 |
| student (Qwen3-8B, first 1M labels only) | 0.816 | 0.903 | 0.038 | 0.893 | 0.444 |
| teacher (Opus 5.5 xhigh, Jev shortlist) | 0.867 | 0.921 | 0.214 | 0.935 | 0.532 |
```

The last command rebuilds the answer key from the judges' raw rows and should report no difference. What is in
`data/` is described in [`data/README.md`](data/README.md).

## 2. Run the released model on the test set (one GPU, minutes)

The weights are GitHub release assets (Apache-2.0); `tools/download.py` fetches them and checks every sha256 against
[`models/MANIFEST.json`](models/MANIFEST.json).

```bash
python3 tools/download.py model                     # -> topic-classifier-v2/ (15 GB)
python3 tools/fetch_works.py --ids data/gold/test_works.jsonl --out work/test_texts.jsonl
pip install torch "transformers>=4.51,<5" safetensors
python3 student/infer.py --model topic-classifier-v2 --input work/test_texts.jsonl --out work/test_preds.jsonl.gz
python3 eval/score_file.py work/test_preds.jsonl.gz
```

`student/infer.py` needs about 25 GB of GPU memory in bf16 (an A100, H100, L40S or similar); 2,200 works take under a
minute. `eval/score_file.py` scores any predictions file in this format against the answer key.

What to expect: on the texts the model read in October 2026, the released bf16 weights give 81.8% on random works
(field 89.6%, calibration error 0.047) and 89.3% on highly cited ones, with the same top topic as the training run's
outputs on 99.6% of works. Texts fetched today differ a little (abstracts added or corrected, records merged), so expect
results within about a point of that.

**The production path.** OpenAlex tags the corpus with [`student/score_vllm.py`](student/score_vllm.py): vLLM, FP8
weights, about 300 works a second on one H100 (`pip install "vllm>=0.10,<0.11" "transformers>=4.53,<4.56"`).

```bash
python3 student/score_vllm.py --model topic-classifier-v2 --input work/test_texts.jsonl --out work/test_fp8.jsonl.gz --top 20
python3 eval/score_file.py work/test_fp8.jsonl.gz
```

FP8_RESULTS

## Tag your own works

Any JSON lines file with `id`, `title`, `abstract` and `venue` works as input to either script. Each output row is
`{"id": ..., "top": [[topic id, probability], ...]}`, highest first; `NOT_CLASSIFIABLE` means the record has nothing to
classify. OpenAlex serves a work's top three topics, with their probabilities as scores, and serves no topic when the
top answer is `NOT_CLASSIFIABLE`. The model's input is built by `work_text` in [`student/model.py`](student/model.py):
title, abstract (first 4,000 characters) and venue, cut at 384 tokens.

## 3. Rebuild the pipeline (API keys and GPUs)

Each stage reads the previous stage's released output, so you can start anywhere.

**The answer key.** `gold/judge.py` asks one judge about every gold work (an Opus 5.5 refusal is re-asked with `--judge opus5`) (needs `pip install anthropic httpx`,
`ANTHROPIC_API_KEY`, and `OPENROUTER_API_KEY` for GPT-6.1 Sol). Rebuild the texts first with `tools/fetch_works.py`.

```bash
python3 tools/fetch_works.py --ids data/gold/test_works.jsonl --out work/test_texts.jsonl
python3 gold/judge.py --judge opus --works work/test_texts.jsonl --gold-works data/gold/test_works.jsonl --out work/judges/test_opus.jsonl.gz
python3 gold/judge.py --judge sol  --works work/test_texts.jsonl --gold-works data/gold/test_works.jsonl --out work/judges/test_sol.jsonl.gz
python3 gold/build_gold.py --split test --judges work/judges --out work/panel_gold_test.jsonl --need-fable work/need_fable.json
python3 gold/judge.py --judge fable --works work/test_texts.jsonl --gold-works data/gold/test_works.jsonl --out work/judges/test_fable.jsonl.gz --ids-file work/need_fable.json
python3 gold/build_gold.py --split test --judges work/judges --out work/panel_gold_test.jsonl
```

Judges are not deterministic, so a rerun gives a slightly different key. Ours agreed on the exact topic for 75% of
random works; expect a similar rate.

**The shortlists.** `shortlist/retriever.py` ranks topics with the released retriever (`tools/download.py retriever`;
`pip install sentence-transformers`) and retrains it from `retriever_training_labels.jsonl.gz`.
`shortlist/jev_shortlist.py` holds the exact Jev request and the shortlist rule; calling Jev needs a key from
[TypeSafe AI](https://typesafe.ai). Jev's answers for the gold works are in `data/shortlist/` and for the first million
in the release asset `jev_shortlists_first_million.jsonl.gz`.

**The teacher.** `teacher/teacher.py` runs one teacher arm on the gold works (rows like `data/teacher_arms/`). For the
2M labels: fetch the texts of the labelled works (`tools/fetch_works.py --ids teacher_labels_2m.jsonl.gz`, about 40,000
API requests), then

```bash
python3 teacher/build_batches.py --works work/train_texts.jsonl --labels teacher_labels_2m.jsonl.gz --out work/req
python3 teacher/submit_batches.py --dir work/req      # Message Batches API; resumable
python3 teacher/collect.py --dir work/req
python3 teacher/fallback.py --dir work/req           # Opus 5, then GPT-6 Astra, for what Opus 5.5 declined
```

`build_batches.py` sends each work the same candidates the released label was chosen from. The released labels say
which model made each one (`model`), which shortlist it came from, and for the second million which rare topic it was
picked to fill. How the second million was picked is in `teacher/select_tail.py`.

**The student.** Build the training file from the labels and the texts, put it on a Modal Volume and train on 8 GPUs
([`student/modal_train.py`](student/modal_train.py); any 8-GPU machine with `accelerate launch --multi_gpu` works):

```bash
python3 student/build_training_file.py --labels teacher_labels_2m.jsonl.gz --texts work/train_texts.jsonl --out work/train/t2m_train.jsonl.gz
# plus work/train/t2m_dev.jsonl and t2m_test.jsonl ({id, ti, ab, ve}) and work/train/taxonomy.json (= data/topics/topics.json)
modal volume put topic-student work/train data
modal run student/modal_train.py --tag my_q8b --backbone Qwen/Qwen3-8B --gpu H200 \
  --extra "--maxlen 384 --epochs 1 --lr 1e-5 --head_lr 1e-3 --bs 8 --optim8bit --grad_ckpt --prefix t2m_"
```

The released model took 6.7 hours on 8 H200s. Training writes the model's logits on the development and test sets and
on 20,000 held-out training works; `eval/students.py` scores those the same way as the released ones. A retrain on
texts fetched today will land close to, not exactly on, the released numbers.

## What cannot be redone from outside

The samples were drawn with SQL over OpenAlex's internal tables (the queries are described in
[benchmarks/](benchmarks/README.md#the-answer-key) and [`data/README.md`](data/README.md)); the files here hold every id
they produced, which is all any later step needs. The models we called (Claude, GPT, Jev) will change over time;
their answers as given are all saved here.
