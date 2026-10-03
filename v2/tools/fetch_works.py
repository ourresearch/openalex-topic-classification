"""Rebuild the text of any works from the public OpenAlex API: the repo holds work ids, never titles or abstracts.
Writes JSON lines {id, title, abstract, venue, year, type, language} (abstract rebuilt from abstract_inverted_index,
venue = the primary location's source name). Resumable: skips ids already in --out.
Usage: python tools/fetch_works.py --ids data/gold/test_works.jsonl --out work/test_texts.jsonl [--api-key KEY]
--ids takes JSON lines with an "id" field (gzip ok) or a text file with one id per line.
Texts come from today's OpenAlex, so a few will differ from what the models read in September and October 2026."""
import argparse, gzip, json, os, time, urllib.parse, urllib.request

SELECT = "id,title,abstract_inverted_index,primary_location,publication_year,type,language"


def abstract(ii):
    if not ii:
        return None
    pos = {p: w for w, ps in ii.items() for p in ps}
    return " ".join(pos[k] for k in sorted(pos))


def get(url, tries=6):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if getattr(e, "code", None) == 404:
                return None
            time.sleep(2 ** k)
    raise RuntimeError(f"failed: {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY")); ap.add_argument("--mailto", default=os.environ.get("OPENALEX_MAILTO"))
    a = ap.parse_args()
    op = gzip.open if a.ids.endswith(".gz") else open
    ids = []
    for line in op(a.ids, "rt"):
        line = line.strip()
        if line:
            ids.append(json.loads(line)["id"] if line.startswith("{") else line)
    ids = [i.rsplit("/", 1)[-1] for i in ids]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8")}
    todo = [i for i in ids if i not in done]
    extra = {k: v for k, v in (("api_key", a.api_key), ("mailto", a.mailto)) if v}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as out:
        for s in range(0, len(todo), 50):
            batch = todo[s:s + 50]
            q = urllib.parse.urlencode({"filter": "ids.openalex:" + "|".join(batch), "per-page": 50, "select": SELECT, **extra})
            got = {w["id"].rsplit("/", 1)[-1]: w for w in get(f"https://api.openalex.org/works?{q}")["results"]}
            for wid in batch:
                w = got.get(wid)
                if w is None:   # the ids filter can miss recently added ids; a single lookup finds them
                    w = get(f"https://api.openalex.org/works/{wid}?" + urllib.parse.urlencode({"select": SELECT, **extra}))
                if w is None:
                    out.write(json.dumps({"id": wid, "missing": True}) + "\n"); continue
                pl = w.get("primary_location") or {}; src = pl.get("source") or {}
                out.write(json.dumps({"id": wid, "title": w.get("title"), "abstract": abstract(w.get("abstract_inverted_index")),
                                      "venue": src.get("display_name") or pl.get("raw_source_name"), "year": w.get("publication_year"),
                                      "type": w.get("type"), "language": w.get("language")}, ensure_ascii=False) + "\n")
            out.flush()
    print(f"{len(ids):,} ids; fetched {len(todo):,} -> {a.out}")


if __name__ == "__main__":
    main()
