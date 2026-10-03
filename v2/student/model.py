"""The student: Qwen3-8B with a linear head over 4,517 classes (the 4,516 OpenAlex topics in data/topics/topics.json
order, then NOT_CLASSIFIABLE). It reads one text per work, built by `work_text`, truncated on the right at 384 tokens,
and pools the last token's hidden state. Probabilities are a plain softmax over the head's logits."""
import json, os

MAX_TOKENS = 384
NOT_CLASSIFIABLE = 4516
HERE = os.path.dirname(os.path.abspath(__file__))
TOPICS = json.load(open(os.path.join(HERE, "..", "data", "topics", "topics.json"), encoding="utf-8"))
CLASS_IDS = [t["id"] for t in TOPICS] + ["NOT_CLASSIFIABLE"]


def work_text(title, abstract, venue):
    """The exact input the student was trained on: title, abstract (first 4,000 characters, omitted when empty), venue."""
    parts = [f"Title: {title or '(none)'}"]
    if abstract:
        parts.append(f"Abstract: {abstract[:4000]}")
    parts.append(f"Venue: {venue or '(unknown)'}")
    return "\n".join(parts)


def load(model_dir, device="cuda", dtype="bfloat16"):
    """-> (tokenizer, encoder, head). model_dir is the unpacked release folder (config, tokenizer, model-*.safetensors,
    head.safetensors)."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer
    dt = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = AutoModel.from_pretrained(model_dir, torch_dtype=dt).to(device).eval()
    sd = load_file(os.path.join(model_dir, "head.safetensors"))
    head = torch.nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0]).to(device)
    head.load_state_dict(sd)
    head.eval()
    return tok, enc, head


def probabilities(tok, enc, head, texts):
    """Softmax over the 4,517 classes for a batch of texts (float32 tensor on the model's device)."""
    import torch
    t = tok(texts, truncation=True, max_length=MAX_TOKENS, padding="longest", return_tensors="pt")
    ids, mask = t["input_ids"].to(enc.device), t["attention_mask"].to(enc.device)
    with torch.no_grad():
        h = enc(input_ids=ids, attention_mask=mask).last_hidden_state
        last = h[torch.arange(h.shape[0], device=h.device), mask.sum(1) - 1].float()
        return torch.softmax(head(last), 1)
