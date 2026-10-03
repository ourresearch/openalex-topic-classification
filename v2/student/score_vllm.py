"""Tag works with vLLM: the path OpenAlex runs over the whole corpus (FP8 by default, about 300 works a second on one
H100). vLLM serves the encoder as a pooling model (last token, no normalisation); the head is applied on the GPU.
vLLM needs a causal-LM checkpoint, so the first run writes one next to the weights (<model>/vllm/): the released
encoder weights under Qwen3ForCausalLM, with an unused lm_head copied from the input embeddings.
Tested with vllm 0.10.x and transformers 4.53 to 4.55.
Usage: python student/score_vllm.py --model topic-classifier-v2 --input works.jsonl --out preds.jsonl.gz [--quant fp8|none]"""
import argparse, gzip, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CLASS_IDS, MAX_TOKENS, work_text


def causal_checkpoint(model_dir):
    """Write <model_dir>/vllm/ once: the same weights, renamed for Qwen3ForCausalLM (bf16)."""
    import shutil, torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    out = os.path.join(model_dir, "vllm")
    if os.path.exists(os.path.join(out, "config.json")):
        return out
    os.makedirs(out, exist_ok=True)
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    wmap, emb = {}, None
    for f in sorted(set(idx.values())):
        with safe_open(os.path.join(model_dir, f), "pt") as h:
            t = {"model." + n: h.get_tensor(n) for n in h.keys()}
        if "model.embed_tokens.weight" in t:
            emb = t["model.embed_tokens.weight"]
        save_file(t, os.path.join(out, f), metadata={"format": "pt"})
        wmap.update({n: f for n in t})
    save_file({"lm_head.weight": emb.clone()}, os.path.join(out, "lm_head.safetensors"), metadata={"format": "pt"})
    wmap["lm_head.weight"] = "lm_head.safetensors"
    json.dump({"metadata": {}, "weight_map": wmap}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    cfg.update(architectures=["Qwen3ForCausalLM"], tie_word_embeddings=False)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    for f in os.listdir(model_dir):
        if f.startswith(("tokenizer", "vocab", "merges", "special_tokens", "added_tokens", "chat_template")):
            shutil.copy(os.path.join(model_dir, f), out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--quant", default="fp8"); ap.add_argument("--top", type=int, default=10); ap.add_argument("--chunk", type=int, default=20000)
    a = ap.parse_args()
    ckpt = causal_checkpoint(a.model)   # before vLLM starts: its engine process must not be forked after this work
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    import numpy as np, torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(ckpt)
    kw = dict(model=ckpt, dtype="bfloat16", max_model_len=512, gpu_memory_utilization=0.85, enable_prefix_caching=False)
    if a.quant and a.quant != "none":
        kw["quantization"] = a.quant
    pooler = {"pooling_type": "LAST", "normalize": False}
    try:
        llm = LLM(runner="pooling", convert="embed", override_pooler_config=pooler, **kw)
    except TypeError:   # older vLLM
        llm = LLM(task="embed", override_pooler_config=pooler, **kw)
    sd = load_file(os.path.join(a.model, "head.safetensors"))
    head = torch.nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0]).cuda(); head.load_state_dict(sd); head.eval()
    op = gzip.open if a.input.endswith(".gz") else open
    rows = [json.loads(l) for l in op(a.input, "rt", encoding="utf-8") if l.strip()]
    t0 = time.time()
    ids = tok([work_text(r.get("title"), r.get("abstract"), r.get("venue")) for r in rows], truncation=True, max_length=MAX_TOKENS)["input_ids"]
    w = gzip.open(a.out, "wt") if a.out.endswith(".gz") else open(a.out, "w")
    with w:
        for s in range(0, len(rows), a.chunk):
            outs = llm.embed([TokensPrompt(prompt_token_ids=x) for x in ids[s:s + a.chunk]], use_tqdm=False)
            H = torch.tensor(np.array([o.outputs.embedding for o in outs]), dtype=torch.float32, device="cuda")
            with torch.no_grad():
                v, k = torch.softmax(head(H), 1).topk(a.top, 1)
            for r, vv, kk in zip(rows[s:s + a.chunk], v.tolist(), k.tolist()):
                w.write(json.dumps({"id": r["id"], "top": [[CLASS_IDS[c], round(p, 6)] for c, p in zip(kk, vv)]}) + "\n")
    print(f"{len(rows):,} works in {time.time() - t0:.0f}s ({len(rows) / (time.time() - t0):.0f}/s) -> {a.out}")


if __name__ == "__main__":
    main()
