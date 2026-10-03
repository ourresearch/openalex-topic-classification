# The student

The released model: [Qwen3-8B](https://github.com/QwenLM/Qwen3) with a linear head over 4,517 classes (the 4,516
topics in [`data/topics/topics.json`](../data/topics/topics.json) order, then `NOT_CLASSIFIABLE`). It reads one text per
work (`work_text` in [`model.py`](model.py): title, abstract, venue), cut on the right at 384 tokens, takes the last
token's hidden state and applies the head; the probabilities are a plain softmax.

| File | What it does |
|---|---|
| [`model.py`](model.py) | The input text, loading the weights, the probabilities |
| [`infer.py`](infer.py) | Tag a file of works with Hugging Face transformers (one GPU, bf16) |
| [`score_vllm.py`](score_vllm.py) | Tag with vLLM in FP8, the path OpenAlex runs over the corpus |
| [`build_training_file.py`](build_training_file.py) | Teacher labels + texts -> the training file |
| [`train.py`](train.py) | Training (multi-GPU with Accelerate); every flag used in the ablations |
| [`modal_train.py`](modal_train.py) | `train.py` on 8 GPUs on Modal |

**Weights.** `python3 tools/download.py model` fetches `topic-classifier-v2/`: the fine-tuned Qwen3-8B encoder in bf16
(`model-0000N-of-00009.safetensors`, a `Qwen3Model` checkpoint loadable with `AutoModel`), the tokenizer, and
`head.safetensors` (`weight` 4517 x 4096, `bias` 4517, float32). Training kept float32 weights; the release is bf16,
which gives the same top topic on 99.6% of the test works and the same accuracy to within 0.1 point.

**Recipe.** One epoch over 1,799,905 works: the 2,000,328 labels from both millions other than `NONE_FIT`, minus the
200,423 held-out works whose id ends in 7, cross-entropy on the teacher's answer, learning rate 1e-5 for the backbone and 1e-3
for the head with 3% warm-up and linear decay, batch 8 per GPU on 8 H200s, 8-bit AdamW, gradient checkpointing, bf16
autocast, seed 1485. 6.7 hours.
