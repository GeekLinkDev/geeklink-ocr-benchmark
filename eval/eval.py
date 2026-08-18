#!/usr/bin/env python3
"""Score OCR predictions against the benchmark ground truth.

Usage:
    python3 eval/eval.py --pred your_predictions.csv
    python3 eval/eval.py --pred your_predictions.jsonl
    python3 eval/eval.py --pred baselines/geeklink.csv

Your predictions file must have one row per sample `id` (matching
data/manifest.csv) and a `prediction` field with the recognized text.
CSV or JSONL both work; unmatched ids are skipped and reported.

Metrics: CER (character error rate) and WER (word error rate) via
Levenshtein edit distance, overall and broken down by language and by
whether the sample has a nearby synthetic watermark (data/manifest.csv
`has_watermark` column) — the watermark subset is the harder detection
case: an engine that dumps every detected text line into `prediction`
without filtering will score much worse there.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "..", "data", "manifest.csv")


def edit_distance(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def load_manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def load_predictions(path):
    preds = {}
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                preds[d["id"]] = d.get("prediction", "")
    else:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                preds[row["id"]] = row.get("prediction", "")
    return preds


def score(manifest, preds, key_fn, label_col, label_width=8):
    buckets = defaultdict(lambda: {"cer_num": 0, "cer_den": 0, "wer_num": 0, "wer_den": 0, "n": 0})
    overall = buckets["__all__"]
    for sid, row in manifest.items():
        if sid not in preds:
            continue
        ref = row["ground_truth"]
        pred = preds[sid]
        key = key_fn(row)
        for bucket in (buckets[key], overall):
            bucket["cer_num"] += edit_distance(pred, ref)
            bucket["cer_den"] += max(len(ref), 1)
            bucket["wer_num"] += edit_distance(pred.split(), ref.split())
            bucket["wer_den"] += max(len(ref.split()), 1)
            bucket["n"] += 1

    print(f"\n-- by {label_col} --")
    print(f"{label_col:<{label_width}}{'n':>6}{'CER':>10}{'WER':>10}")
    for k in sorted(x for x in buckets if x != "__all__"):
        b = buckets[k]
        print(f"{k:<{label_width}}{b['n']:>6}{b['cer_num'] / b['cer_den']:>10.4f}{b['wer_num'] / b['wer_den']:>10.4f}")
    b = overall
    print(f"{'ALL':<{label_width}}{b['n']:>6}{b['cer_num'] / b['cer_den']:>10.4f}{b['wer_num'] / b['wer_den']:>10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Predictions file (CSV or JSONL) with id,prediction columns")
    args = ap.parse_args()

    manifest = load_manifest()
    preds = load_predictions(args.pred)

    missing = [sid for sid in manifest if sid not in preds]
    if missing:
        print(f"warning: {len(missing)} sample(s) missing from predictions, skipped", file=sys.stderr)

    score(manifest, preds, lambda row: row["lang"], "lang")
    score(manifest, preds, lambda row: "watermark" if row["has_watermark"] == "True" else "clean",
          "watermark?", label_width=10)


if __name__ == "__main__":
    main()
