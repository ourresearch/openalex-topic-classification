"""Train a student (multi-GPU, `accelerate launch --multi_gpu train.py ...`; modal_train.py runs it on 8 GPUs). The
student reads only the work (title, abstract, venue; model.work_text) and scores all 4,516 topics + NOT_CLASSIFIABLE
(class 4516) with cross-entropy on the teacher's labels. Rows with h = true (work id ending in 7) are held out: the first
20,000 of them measure teacher fidelity.
--kind qwen: a decoder (Qwen3-1.7B / 4B / 8B), last-token pooled. --kind e5: an encoder (the retriever), mean-pooled and
L2-normalised, head initialised from topic embeddings in <data>/topics.npy.
The released model: --kind qwen --backbone Qwen/Qwen3-8B --maxlen 384 --epochs 1 --lr 1e-5 --head_lr 1e-3 --bs 8
--optim8bit --grad_ckpt on 8x H200 (6.7 h). The other flags are the ablations in eval/students.py.
The learning-rate schedule is stepped by hand once per optimizer step (Accelerate's prepared scheduler steps once per
process). Data files in --data: <prefix>train.jsonl.gz rows {id, ti, ab, ve, y, s, h} (build_training_file.py),
<prefix>dev.jsonl and <prefix>test.jsonl rows {id, ti, ab, ve}, taxonomy.json (= data/topics/topics.json).
Writes <out>/models/<tag>/ (enc/, head.pt, args.json) and <out>/results/<tag>_{dev,test,hold}.npz (logits)."""
import argparse, gzip, json, math, os, random, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import AutoTokenizer, AutoModel
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--kind", choices=["e5", "qwen"], required=True); ap.add_argument("--backbone", required=True)
ap.add_argument("--maxlen", type=int, default=256); ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--head_lr", type=float, default=1e-3); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--data", default="/vol/data"); ap.add_argument("--out", default="/vol"); ap.add_argument("--fold", type=int, default=-1)   # 0/1: train on the other half (int(id[1:]) // 10 % 2), shortlist this half
ap.add_argument("--topk", type=int, default=30)
ap.add_argument("--task", choices=["cls", "picker"], default="cls")   # picker: rows carry c = shortlist (topic idx), y = position 1..K or 0 = not classifiable
ap.add_argument("--prefix", default="")
ap.add_argument("--pool", choices=["last", "mean"], default="last"); ap.add_argument("--soft", type=float, default=0.0)   # mass on teacher secondaries
ap.add_argument("--ls", type=float, default=0.0); ap.add_argument("--head_init", choices=["none", "text"], default="none"); ap.add_argument("--aux", type=float, default=0.0)   # subfield + field CE weight
ap.add_argument("--bucket", action="store_true")   # length-bucketed batches: shuffle, sort pools of 64 batches by length, shuffle the batches
ap.add_argument("--optim8bit", action="store_true"); ap.add_argument("--grad_ckpt", action="store_true")   # 8B on H200: fp32 weights + 8-bit AdamW + checkpointing   # data files <prefix>train.jsonl.gz, <prefix>dev.jsonl, <prefix>test.jsonl
a = ap.parse_args()
OPT = None
if a.task == "picker":
    TAX = json.load(open(f"{a.data}/taxonomy.json")); OPT = [f"{t['display_name']}: {', '.join((t.get('keywords') or [])[:5])}" for t in TAX]
TAXL = json.load(open(f"{a.data}/taxonomy.json"))
SUBS = sorted({t["subfield"]["id"] for t in TAXL}); FLDS = sorted({t["field"]["id"] for t in TAXL})
SUB_OF = torch.tensor([SUBS.index(t["subfield"]["id"]) for t in TAXL] + [len(SUBS)]); FLD_OF = torch.tensor([FLDS.index(t["field"]["id"]) for t in TAXL] + [len(FLDS)])   # NONE -> extra class
NC = 4517 if a.task == "cls" else a.topk + 1; acc = Accelerator(mixed_precision="bf16", kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False, gradient_as_bucket_view=True)]); rank, world = acc.process_index, acc.num_processes; torch.manual_seed(1485)
def log(*x):
    if acc.is_main_process: print(*x, flush=True)
def text(r):
    if a.kind == "e5": return f"query: {r['ti']}. {(r['ab'] or '')[:2000]} Venue: {r['ve']}"
    p = [f"Title: {r['ti'] or '(none)'}"]
    if r["ab"]: p.append(f"Abstract: {r['ab'][:4000]}")
    p.append(f"Venue: {r['ve'] or '(unknown)'}")
    if a.task == "picker": p.append("\nCandidate topics:\n" + "\n".join(f"{j + 1}. {OPT[c]}" for j, c in enumerate(r["c"][:a.topk])))
    return "\n".join(p)
rows = []
with gzip.open(f"{a.data}/{a.prefix}train.jsonl.gz", "rt") as fh:
    for l in fh:
        rows.append(json.loads(l))
        if a.limit and len(rows) >= a.limit: break
def fold(r): return int(r["id"][1:]) // 10 % 2
train = [r for r in rows if not r["h"] and (a.fold < 0 or fold(r) != a.fold)]; hold = [r for r in rows if r["h"]][:20000]
short = [r for r in rows if a.fold >= 0 and fold(r) == a.fold]   # cross-fit shortlists for the picker (incl. held-out rows)
log(f"{a.tag}: {len(train):,} train rows, {len(hold):,} held-out, world {world}")
tok = AutoTokenizer.from_pretrained(a.backbone); tok.padding_side = "right"
if tok.pad_token is None: tok.pad_token = tok.eos_token
class Student(nn.Module):
    def __init__(s):
        super().__init__(); s.enc = AutoModel.from_pretrained(a.backbone, torch_dtype=torch.float32); H = s.enc.config.hidden_size
        if getattr(s.enc, "pooler", None) is not None: s.enc.pooler = None   # unused (we pool ourselves); DDP rejects params without grads
        s.head = nn.Linear(H, NC)
        if a.aux: s.sub = nn.Linear(H, len(SUBS) + 1); s.fld = nn.Linear(H, len(FLDS) + 1)
        if a.kind == "e5" and a.task == "cls":
            T = torch.tensor(np.load(f"{a.data}/topics.npy"), dtype=torch.float32)
            with torch.no_grad(): s.head.weight.zero_(); s.head.weight[:T.shape[0]] = T * 20.0; s.head.bias.zero_()
    def pool(s, ids, mask):
        h = s.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        if a.kind == "e5": return F.normalize((h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True), dim=-1)
        if a.pool == "mean": return (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        return h[torch.arange(h.shape[0], device=h.device), mask.sum(1) - 1]
    def forward(s, ids, mask, aux=False):
        p = s.pool(ids, mask).float(); z = s.head(p)
        return (z, s.sub(p), s.fld(p)) if aux else z
m = Student()
if a.head_init == "text" and a.task == "cls":
    m.to(acc.device); m.eval(); ttxt = [f"Topic: {t['display_name']}. Keywords: {', '.join((t.get('keywords') or [])[:10])}" for t in TAXL]; V = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s0 in range(0, len(ttxt), 64):
            t = tok(ttxt[s0:s0 + 64], truncation=True, max_length=96, padding="longest", return_tensors="pt"); V.append(m.pool(t["input_ids"].to(acc.device), t["attention_mask"].to(acc.device)).float())
    V = torch.cat(V); mu = V.mean(0, keepdim=True); Vc = V - mu
    W0 = Vc / Vc.norm(dim=1, keepdim=True) / (Vc.norm(dim=1).mean()) * 10.0   # logit_i ~ 10 * cos(p - mu, v_i - mu) at step 0
    with torch.no_grad(): m.head.weight[:len(TAXL)] = W0.to(m.head.weight.device); m.head.bias[:len(TAXL)] = -(W0 @ mu[0]).to(m.head.bias.device); m.head.weight[len(TAXL):] = 0; m.head.bias[len(TAXL):] = 0
    m.train(); log(f"head initialised from topic texts ({len(TAXL)} rows)")
if a.grad_ckpt: m.enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
log(f"params {len(list(m.parameters()))}; last: {[n for n, _ in m.named_parameters()][-3:]}")
groups = [{"params": m.enc.parameters(), "lr": a.lr}, {"params": m.head.parameters(), "lr": a.head_lr}]
if a.optim8bit:
    import bitsandbytes as bnb; opt = bnb.optim.AdamW8bit(groups, weight_decay=0.01)
else: opt = torch.optim.AdamW(groups, weight_decay=0.01)
m, opt = acc.prepare(m, opt)
per_proc = len(train) // world // a.bs; total = max(1, int(per_proc * a.epochs)); warm = max(1, int(0.03 * total))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)) if s >= warm else (s + 1) / warm)
def batch(rs):
    t = tok([text(r) for r in rs], truncation=True, max_length=a.maxlen, padding="longest", return_tensors="pt")
    return t["input_ids"].to(acc.device), t["attention_mask"].to(acc.device)
log(f"steps per process per epoch {per_proc:,}, total {total:,}, warmup {warm}")
step, t0, run = 0, time.time(), 0.0; m.train()
while step < total:
    ep = step // per_proc; order = list(range(len(train))); random.Random(1485 + ep).shuffle(order); mine = order[rank::world]
    if a.bucket:   # same shuffle, then sort within pools of 64 batches by text length and shuffle the resulting batches
        pool = 64 * a.bs; ln = lambda i: len(train[i]["ti"]) + len(train[i]["ab"] or "")
        bs_list = []
        for p0 in range(0, len(mine), pool):
            chunk = sorted(mine[p0:p0 + pool], key=ln); bs_list += [chunk[j:j + a.bs] for j in range(0, len(chunk), a.bs)]
        random.Random(7 + ep + 1000 * rank).shuffle(bs_list); mine = [i for b in bs_list for i in b]
    for s in range(0, per_proc * a.bs, a.bs):
        if step >= total: break
        rs = [train[i] for i in mine[s:s + a.bs]]; ids, mask = batch(rs); y = torch.tensor([r["y"] for r in rs], device=acc.device)
        with acc.autocast():
            out = m(ids, mask, aux=a.aux > 0); z = out[0] if a.aux else out
            if a.soft and a.task == "cls":
                tgt = torch.zeros_like(z, dtype=torch.float32)
                for k, r in enumerate(rs):
                    sec = [x for x in r.get("s", []) if x != r["y"]]
                    tgt[k, r["y"]] = 1.0 - (a.soft if sec else 0.0)
                    for x in sec: tgt[k, x] = a.soft / len(sec)
                if a.ls: tgt = tgt * (1 - a.ls) + a.ls / z.shape[1]
                loss = -(tgt * F.log_softmax(z.float(), 1)).sum(1).mean()
            else: loss = F.cross_entropy(z.float(), y, label_smoothing=a.ls)
            if a.aux: loss = loss + a.aux * (F.cross_entropy(out[1].float(), SUB_OF.to(y.device)[y]) + F.cross_entropy(out[2].float(), FLD_OF.to(y.device)[y]))
        acc.backward(loss); acc.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(); step += 1
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 50 == 0 or step == 1:
            el = time.time() - t0; log(f"step {step}/{total} ep {step / per_proc:.2f} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} {el:.0f}s eta {(total - step) * el / step / 60:.0f}m")
acc.wait_for_everyone(); um = acc.unwrap_model(m); um.eval()
if acc.is_main_process:
    os.makedirs(f"{a.out}/models/{a.tag}", exist_ok=True); um.enc.save_pretrained(f"{a.out}/models/{a.tag}/enc"); tok.save_pretrained(f"{a.out}/models/{a.tag}/enc")
    torch.save(um.head.state_dict(), f"{a.out}/models/{a.tag}/head.pt"); json.dump(vars(a), open(f"{a.out}/models/{a.tag}/args.json", "w"))
    os.makedirs(f"{a.out}/results", exist_ok=True)
    def infer(rs, B=128):
        out = []
        with torch.no_grad(), acc.autocast():
            for s in range(0, len(rs), B): ids, mask = batch(rs[s:s + B]); out.append(um(ids, mask).float().cpu())
        return torch.cat(out).numpy()
    for name in ("dev", "test"):
        rs = [json.loads(l) for l in open(f"{a.data}/{a.prefix}{name}.jsonl")]; L = infer(rs)
        np.savez(f"{a.out}/results/{a.tag}_{name}.npz", ids=np.array([r["id"] for r in rs]), logits=L.astype(np.float16))
    L = infer(hold); pred = L.argmax(1); y = np.array([r["y"] for r in hold])
    np.savez(f"{a.out}/results/{a.tag}_hold.npz", ids=np.array([r["id"] for r in hold]), y=y, pred=pred, pmax=torch.softmax(torch.tensor(L), 1).max(1).values.numpy())
    if short:   # cross-fit top-K for the picker: rows this model never trained on
        K = a.topk; ti, ts = [], []
        for s0 in range(0, len(short), 4096):
            L = torch.tensor(infer(short[s0:s0 + 4096])); v, i = L.topk(K, 1); ti.append(i.numpy().astype(np.int16)); ts.append(v.numpy().astype(np.float16))
        np.savez(f"{a.out}/results/{a.tag}_short.npz", ids=np.array([r["id"] for r in short]), top=np.concatenate(ti), score=np.concatenate(ts))
        log(f"cross-fit top-{K} for {len(short):,} rows saved")
    log(f"saved; teacher fidelity on {len(hold):,} held-out rows {(pred == y).mean():.4f}; {time.time() - t0:.0f}s total")
