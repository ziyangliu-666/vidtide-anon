#!/usr/bin/env python3
"""Score a leaderboard submission on a release's public evaluation list.

Usage: python scripts/score_submission.py manifests/M7 my_scores.jsonl
"""
import json
import sys
from pathlib import Path


def auroc(pos, neg):
    allv = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum, i, n = 0.0, 0, len(allv)
    while i < n:
        j = i
        while j + 1 < n and allv[j + 1][0] == allv[i][0]:
            j += 1
        rank_sum += ((i + j) / 2 + 1) * sum(1 for k in range(i, j + 1) if allv[k][1] == 1)
        i = j + 1
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    release, sub = Path(sys.argv[1]), sys.argv[2]
    labels = {}
    for line in open(release / "eval_balanced.jsonl"):
        r = json.loads(line)
        labels[r["id"]] = 1 if r["label"] in (1, "fake") else 0
    scores = {r["id"]: float(r["score"]) for r in map(json.loads, open(sub)) if r["id"] in labels}
    pos = [s for i, s in scores.items() if labels[i] == 1]
    neg = [s for i, s in scores.items() if labels[i] == 0]
    print(f"public list: {len(scores)}/{len(labels)} clips scored, AUROC = {100 * auroc(pos, neg):.2f}")


if __name__ == "__main__":
    main()
