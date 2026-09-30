"""Split each ablation stage's error into bias and spread.

RMSE^2 = bias^2 + SD^2. A stage that only moves bias (same SD, same
correlation) does not improve predictive ability; it only moves the
prediction closer to or further from the metric's preferred offset.

Input: one prediction CSV per stage, columns task, seed, split, bearing,
t, y_true, y_pred (same format as smooth_predictions.py). Every stage
must be evaluated on the same degradation segment.

Usage:
  python tools/bias_spread.py S0=preds_s0.csv S1=preds_s1.csv S3=preds_s3.csv
"""
import sys

import numpy as np
import pandas as pd

from smooth_predictions import score_fn


def stage_stats(df):
    rows = []
    for (task, seed), g in df.groupby(["task", "seed"]):
        e = (g["y_pred"] - g["y_true"]).to_numpy(float)
        rows.append(dict(
            task=task, seed=seed,
            score=score_fn(g["y_true"], g["y_pred"]),
            rmse=float(np.sqrt((e ** 2).mean())),
            bias=float(e.mean()),
            sd=float(e.std()),
            corr=float(np.corrcoef(g["y_true"], g["y_pred"])[0, 1]),
            n=len(g),
        ))
    return pd.DataFrame(rows)


def main():
    stages = [a.split("=", 1) for a in sys.argv[1:]]
    if not stages:
        raise SystemExit(__doc__)
    out, lengths = [], {}
    for name, path in stages:
        df = pd.read_csv(path)
        df = df[df["split"] == "test"] if "split" in df else df
        s = stage_stats(df)
        lengths[name] = s.set_index(["task", "seed"])["n"]
        per_seed = s.groupby("seed")[["score", "rmse", "bias", "sd", "corr"]].mean()
        row = per_seed.mean().rename(name)
        out.append(row)
        print(f"\n{name} per task:")
        print(s.groupby("task")[["score", "rmse", "bias", "sd", "corr"]].mean().round(4))

    ref = next(iter(lengths.values()))
    for name, n in lengths.items():
        if not n.reindex(ref.index).equals(ref):
            print(f"\nWARNING: {name} is evaluated on a different number of samples "
                  "than the first stage; the stages are not comparable.")

    print("\nsix-task mean (averaged per seed):")
    print(pd.DataFrame(out).round(4))


if __name__ == "__main__":
    main()
