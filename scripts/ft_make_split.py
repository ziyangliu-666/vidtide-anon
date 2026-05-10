"""Build deterministic train/test splits for DeMamba LP fine-tune.

Pool: videos that successfully scored under the off-the-shelf demamba_pika
detector (i.e. frame extraction works). Test: 1K fake + 1K real, stratified
by platform. Train: the rest (~4K + ~4K).

Outputs:
  - data/splits/ft_demamba_train.jsonl   {video, label, platform, generator}
  - data/splits/ft_demamba_test.jsonl
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "vidtide.db"
SCORES = REPO / "results" / "bench_5k_demamba_pika_scores.jsonl"
OUT_DIR = REPO / "data" / "splits"
SEED = 42
N_TEST_PER_CLASS = 1000


def main() -> None:
    con = sqlite3.connect(DB)
    meta = {
        vid: (plat, gen or None)
        for vid, plat, gen in con.execute(
            "SELECT id, source_platform, COALESCE(NULLIF(claimed_generator,''),'') FROM videos"
        )
    }
    con.close()

    fakes_by_plat: dict[str, list[dict]] = defaultdict(list)
    reals_by_plat: dict[str, list[dict]] = defaultdict(list)
    with SCORES.open() as f:
        for line in f:
            r = json.loads(line)
            vid = r["video"].replace(".mp4", "")
            m = meta.get(vid)
            if m is None:
                continue
            plat, gen = m
            row = {"video": vid, "label": int(r["label"]), "platform": plat, "generator": gen}
            (fakes_by_plat if r["label"] == 1 else reals_by_plat)[plat].append(row)

    rng = random.Random(SEED)
    for v in fakes_by_plat.values(): rng.shuffle(v)
    for v in reals_by_plat.values(): rng.shuffle(v)

    def stratified_sample(pool: dict[str, list[dict]], n: int) -> tuple[list, list]:
        total = sum(len(v) for v in pool.values())
        test, train = [], []
        for plat, rows in pool.items():
            n_test_plat = max(1, round(n * len(rows) / total))
            test.extend(rows[:n_test_plat])
            train.extend(rows[n_test_plat:])
        # Trim test to exactly n if needed
        rng.shuffle(test)
        if len(test) > n:
            train.extend(test[n:])
            test = test[:n]
        return train, test

    fake_train, fake_test = stratified_sample(fakes_by_plat, N_TEST_PER_CLASS)
    real_train, real_test = stratified_sample(reals_by_plat, N_TEST_PER_CLASS)

    train = fake_train + real_train
    test = fake_test + real_test
    rng.shuffle(train); rng.shuffle(test)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "ft_demamba_train.jsonl").open("w") as f:
        for r in train: f.write(json.dumps(r) + "\n")
    with (OUT_DIR / "ft_demamba_test.jsonl").open("w") as f:
        for r in test: f.write(json.dumps(r) + "\n")

    print(f"train: fake={sum(1 for r in train if r['label']==1)} real={sum(1 for r in train if r['label']==0)}")
    print(f"test:  fake={sum(1 for r in test if r['label']==1)} real={sum(1 for r in test if r['label']==0)}")
    print("\ntest by platform:")
    plat_count: dict = defaultdict(lambda: [0,0])
    for r in test:
        plat_count[r["platform"]][r["label"]] += 1
    for p, (n_real, n_fake) in sorted(plat_count.items()):
        print(f"  {p:<10} fake={n_fake} real={n_real}")


if __name__ == "__main__":
    main()
