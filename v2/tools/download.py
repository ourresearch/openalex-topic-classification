"""Download release assets listed in models/MANIFEST.json, check every sha256, and unpack them.
  python tools/download.py model               # the released model -> topic-classifier-v2/ (15 GB)
  python tools/download.py labels              # the 2M teacher labels and the first million's Jev shortlists
  python tools/download.py shortlist-student   # the Qwen3-1.7B student that shortlisted the second million (3.5 GB)
  python tools/download.py retriever           # the fine-tuned multilingual-e5-base retriever and its training labels
  python tools/download.py all
Options: --dest DIR (default: the current directory). Standard library only; resumes by skipping verified files."""
import argparse, hashlib, json, os, sys, tarfile, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(HERE, "..", "models", "MANIFEST.json")))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fetch(name, meta, dest):
    """Download one asset to its place: <folder>/<file> for weights, the file itself otherwise."""
    folder, _, file = name.partition("__")
    path = os.path.join(dest, folder, file) if file else os.path.join(dest, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) == meta["bytes"] and sha256(path) == meta["sha256"]:
        return path
    print(f"downloading {name} ({meta['bytes'] / 1e9:.2f} GB)", flush=True)
    urllib.request.urlretrieve(M["base_url"] + name, path + ".part")
    got = sha256(path + ".part")
    if got != meta["sha256"]:
        sys.exit(f"sha256 mismatch for {name}: {got}")
    os.replace(path + ".part", path)
    return path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=sorted({a["group"] for a in M["assets"].values()} | {"all"}))
    ap.add_argument("--dest", default=".")
    a = ap.parse_args()
    for name, meta in M["assets"].items():
        if a.what in ("all", meta["group"]):
            p = fetch(name, meta, a.dest)
            if p.endswith(".tar.gz"):
                with tarfile.open(p) as t:
                    try:
                        t.extractall(a.dest, filter="data")
                    except TypeError:   # Python without extraction filters
                        t.extractall(a.dest)
    print("done; every file matches models/MANIFEST.json")


if __name__ == "__main__":
    main()
