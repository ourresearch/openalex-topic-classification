"""Run student/train.py on Modal (https://modal.com) on 8 GPUs, data-parallel. Put the training files on a Modal Volume
first (modal volume put <volume> work/train data/), then e.g. the released model:
  modal run student/modal_train.py --tag my_q8b --backbone Qwen/Qwen3-8B --gpu H200 --extra "--maxlen 384 --epochs 1 --lr 1e-5 --bs 8 --optim8bit --grad_ckpt"
  modal run student/modal_train.py --tag my_q17 --backbone Qwen/Qwen3-1.7B --gpu H100 --extra "--maxlen 384 --epochs 1 --lr 2e-5 --bs 16"
Outputs land on the Volume under models/<tag>/ and results/<tag>_*.npz; the log under results/train_<tag>.log."""
import os, subprocess, sys, time
import modal

VOLUME = os.environ.get("VOLUME", "topic-student")
app = modal.App("topic-student-train")
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0", "transformers>=4.51,<5.0", "accelerate>=1.0", "numpy", "safetensors", "sentencepiece", "protobuf", "bitsandbytes>=0.45")
         .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py"), "/root/train.py"))


def _train(tag, backbone, kind, extra):
    vol.reload(); os.makedirs("/vol/results", exist_ok=True); log = f"/vol/results/train_{tag}.log"
    cmd = [sys.executable, "-m", "accelerate.commands.launch", "--multi_gpu", "--num_processes", "8", "--mixed_precision", "bf16", "/root/train.py",
           "--tag", tag, "--kind", kind, "--backbone", backbone, "--data", "/vol/data", "--out", "/vol"] + extra.split()
    last = 0
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            f.write(line); f.flush(); print(line.rstrip()[:300], flush=True)
            if time.time() - last > 60: vol.commit(); last = time.time()
    vol.commit(); return p.wait()


@app.function(image=image, gpu="H100:8", volumes={"/vol": vol}, timeout=8 * 3600, memory=262144, cpu=32)
def train_h100(tag: str, backbone: str, kind: str = "qwen", extra: str = ""): return _train(tag, backbone, kind, extra)


@app.function(image=image, gpu="H200:8", volumes={"/vol": vol}, timeout=8 * 3600, memory=524288, cpu=32)   # 4B and 8B: Adam states need > 80 GB
def train_h200(tag: str, backbone: str, kind: str = "qwen", extra: str = ""): return _train(tag, backbone, kind, extra)


@app.local_entrypoint()
def main(tag: str, backbone: str, gpu: str = "H100", kind: str = "qwen", extra: str = ""):
    print((train_h200 if gpu == "H200" else train_h100).remote(tag, backbone, kind, extra))
