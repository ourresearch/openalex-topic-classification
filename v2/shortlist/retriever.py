"""The retriever: multilingual-e5-base fine-tuned contrastively (work text -> topic text) on 72,267 confident Jev
answers. It ranks all 4,516 topics for a work by cosine similarity; its top 255 are the candidates Jev scores.
  encode:  python shortlist/retriever.py --model retriever-e5-base --works work/texts.jsonl --out work/cands255.jsonl
  train:   python shortlist/retriever.py --train --works work/retriever_texts.jsonl --labels retriever_training_labels.jsonl.gz --save my-retriever
Training pairs: every work in the labels file whose Jev top answer has p >= 0.5 (1 epoch, 128 tokens, batch 32,
MultipleNegativesRankingLoss, 20 warm-up steps). Needs `pip install sentence-transformers`."""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
from common import TOPICS, TOPIC, TOPIC_IDS, read_jsonl

PFX = "query: "
def topic_text(t): return PFX + f"{t['display_name']}. Keywords: {', '.join(t.get('keywords') or [])}. {t.get('description') or ''}"
def work_text(w): return PFX + f"{w.get('title') or ''}. {(w.get('abstract') or '')[:2000]} Venue: {w.get('venue') or ''}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-base"); ap.add_argument("--works", required=True); ap.add_argument("--out")
    ap.add_argument("--k", type=int, default=255); ap.add_argument("--train", action="store_true"); ap.add_argument("--labels"); ap.add_argument("--save")
    a = ap.parse_args()
    import numpy as np, torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    works = [w for w in read_jsonl(a.works) if not w.get("missing")]
    if a.train:
        from sentence_transformers import InputExample, losses
        from torch.utils.data import DataLoader
        W = {w["id"]: w for w in works}; ex = []
        for r in read_jsonl(a.labels):
            if r["id"] in W and r["ranked"] and r["ranked"][0]["p"] >= 0.5:
                ex.append(InputExample(texts=[work_text(W[r["id"]]), topic_text(TOPIC[r["ranked"][0]["topic_id"]])]))
        random.Random(0).shuffle(ex); print(f"{len(ex):,} training pairs", flush=True)
        m = SentenceTransformer("intfloat/multilingual-e5-base", device=dev); m.max_seq_length = 128
        m.fit(train_objectives=[(DataLoader(ex, shuffle=True, batch_size=32), losses.MultipleNegativesRankingLoss(m))], epochs=1, warmup_steps=20)
        m.save(a.save); print("saved", a.save); return
    m = SentenceTransformer(a.model, device=dev); m.max_seq_length = 256
    T = m.encode([topic_text(t) for t in TOPICS], batch_size=64, normalize_embeddings=True)
    with open(a.out, "w") as f:
        for s in range(0, len(works), 4096):
            b = works[s:s + 4096]; Wv = m.encode([work_text(w) for w in b], batch_size=64, normalize_embeddings=True)
            S = Wv @ T.T
            for w, row in zip(b, S):
                f.write(json.dumps({"id": w["id"], "candidates": [TOPIC_IDS[j] for j in np.argsort(-row)[:a.k]]}) + "\n")
    print(f"{len(works):,} works -> {a.out}")


if __name__ == "__main__":
    main()
