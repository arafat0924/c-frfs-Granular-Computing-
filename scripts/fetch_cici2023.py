"""Fetch CIC-IoT2023 (CSV feature files) as a per-class sample, without keeping the 9.3 GB of raw files.

Source: public Hugging Face copy bencorn/CIC-IoT-2023 (NOT the official UNB download, which needs a form):
the 63 files CSV/MERGED_CSV/Merged01..63.csv. Each file is downloaded, streamed once, and deleted.
Per label we keep a uniform reservoir sample of at most K rows (seed 0) and the exact row count, so a
loader can still recover the original class proportions.

Run:  .venv/bin/python scripts/fetch_cici2023.py [--out data/cici2023] [--k 20000]
Writes: CICIoT2023_sample.csv, label_counts.json, SOURCE.txt in --out.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import time
import urllib.request

REPO = "bencorn/CIC-IoT-2023"
FILE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/CSV/MERGED_CSV/Merged{{:02d}}.csv"
API_URL = f"https://huggingface.co/api/datasets/{REPO}?blobs=true"
MIN_FREE = 1_500_000_000          # this server is ~95% full: never fill it


def expected_sizes():
    with urllib.request.urlopen(API_URL, timeout=60) as r:
        meta = json.load(r)
    return {s["rfilename"].split("/")[-1]: s["size"] for s in meta["siblings"] if "/MERGED_CSV/" in s["rfilename"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cici2023")
    ap.add_argument("--k", type=int, default=20000)
    ap.add_argument("--files", type=int, default=63)
    args = ap.parse_args()

    tmp = os.path.join(args.out, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    sizes = expected_sizes()
    rng = random.Random(0)
    header = None
    seen, kept = {}, {}                         # label -> rows seen / reservoir of raw lines
    t0 = time.time()

    for i in range(1, args.files + 1):
        name = f"Merged{i:02d}.csv"
        if shutil.disk_usage(args.out).free < MIN_FREE:
            raise SystemExit(f"stopping: less than {MIN_FREE / 1e9:.1f} GB free")
        path = os.path.join(tmp, name)
        subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", path, FILE_URL.format(i)], check=True)
        if os.path.getsize(path) != sizes[name]:
            raise SystemExit(f"{name}: got {os.path.getsize(path)} bytes, expected {sizes[name]}")
        with open(path, encoding="utf-8") as f:
            h = f.readline().rstrip("\r\n")
            if header is None:
                header = h
            elif h != header:
                raise SystemExit(f"{name}: header differs from Merged01.csv")
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                label = line.rsplit(",", 1)[1]
                n = seen[label] = seen.get(label, 0) + 1
                res = kept.setdefault(label, [])
                if len(res) < args.k:
                    res.append(line)
                else:
                    j = rng.randrange(n)
                    if j < args.k:
                        res[j] = line
        os.remove(path)
        print(f"{name} done | rows so far {sum(seen.values()):,} | labels {len(seen)} | {time.time() - t0:.0f}s", flush=True)

    rows = [l for res in kept.values() for l in res]
    rng.shuffle(rows)
    with open(os.path.join(args.out, "CICIoT2023_sample.csv"), "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")
    with open(os.path.join(args.out, "label_counts.json"), "w") as f:
        json.dump({k: dict(rows=seen[k], kept=len(kept[k])) for k in sorted(seen)}, f, indent=2)
    with open(os.path.join(args.out, "SOURCE.txt"), "w") as f:
        f.write(f"Source: https://huggingface.co/datasets/{REPO} (third-party copy, license tag 'cc'; not the official UNB download)\n"
                f"Files: CSV/MERGED_CSV/Merged01..{args.files:02d}.csv, fetched {time.strftime('%Y-%m-%d')}\n"
                f"Sample: uniform reservoir of at most {args.k} rows per label, seed 0; exact counts in label_counts.json\n"
                f"Rows seen: {sum(seen.values())}, rows kept: {len(rows)}, labels: {len(seen)}\n"
                "Official page: https://www.unb.ca/cic/datasets/iotdataset-2023.html\n"
                "Cite: Neto et al., CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment, "
                "Sensors 2023, 23(13), 5941.\n")
    shutil.rmtree(tmp)
    print(f"wrote {len(rows):,} sampled rows from {sum(seen.values()):,} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
