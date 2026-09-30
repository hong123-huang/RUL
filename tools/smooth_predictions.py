"""Causal temporal smoothing of RUL predictions (Ren et al., IEEE Access 2018).

Uses only each bearing's own past predictions, never target labels.
The smoothing parameter must be chosen on source-validation bearings
(--select-split), then applied unchanged to the test split.

Input CSV columns: task, seed, split, bearing, t, y_true, y_pred
  split: e.g. "src_val" / "test"; t: sample index in time order.

Usage:
  python tools/smooth_predictions.py preds.csv --select-split src_val --test-split test
"""
import argparse

import numpy as np
import pandas as pd


def score_fn(y_true, y_pred):
    # Replace with the Score definition used in the paper so numbers match
    # the progress report. Default: PHM2012-style asymmetric accuracy on
    # normalized RUL, underestimation penalized less than overestimation.
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = y_true > 1e-6
    er = 100.0 * (y_true[mask] - y_pred[mask]) / y_true[mask]
    a = np.empty_like(er)
    neg = er <= 0
    a[neg] = np.exp(-np.log(0.5) * er[neg] / 5)
    a[~neg] = np.exp(np.log(0.5) * er[~neg] / 20)
    return float(a.mean())


def ema(x, alpha):
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def causal_ma(x, w):
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0.0))
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - w, 0)
    return (c[idx] - c[lo]) / (idx - lo)


SMOOTHERS = {
    "none": [None],
    "ema": [0.5, 0.3, 0.2, 0.1, 0.05],
    "ma": [3, 5, 10, 20, 40],
}


def apply(df, kind, p):
    df = df.sort_values("t").copy()
    if kind == "none":
        return df
    parts = []
    for _, g in df.groupby(["task", "seed", "split", "bearing"], sort=False):
        g = g.sort_values("t").copy()
        x = g["y_pred"].to_numpy(float)
        g["y_pred"] = ema(x, p) if kind == "ema" else causal_ma(x, int(p))
        parts.append(g)
    return pd.concat(parts)


def metrics(df):
    rows = []
    for (task, seed), g in df.groupby(["task", "seed"]):
        err = g["y_pred"] - g["y_true"]
        rows.append(dict(task=task, seed=seed, score=score_fn(g["y_true"], g["y_pred"]),
                         rmse=float(np.sqrt((err ** 2).mean())), mae=float(err.abs().mean())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--select-split", default="src_val")
    ap.add_argument("--test-split", default="test")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)

    sel = df[df["split"] == args.select_split]
    if sel.empty:
        raise SystemExit(f"no rows with split == {args.select_split!r}; "
                         "the smoothing parameter must not be chosen on test data")
    cands = [(k, p) for k, ps in SMOOTHERS.items() for p in ps]
    sel_scores = {(k, p): metrics(apply(sel, k, p))["score"].mean() for k, p in cands}
    best = max(sel_scores, key=sel_scores.get)
    print("selection split scores:")
    for kp, s in sorted(sel_scores.items(), key=lambda kv: -kv[1]):
        print(f"  {kp[0]:>4} {str(kp[1]):>5}  {s:.4f}")
    print(f"chosen on {args.select_split}: {best}")

    test = df[df["split"] == args.test_split]
    base = metrics(apply(test, "none", None))
    smooth = metrics(apply(test, *best))
    m = base.merge(smooth, on=["task", "seed"], suffixes=("_base", "_smooth"))
    for k in ["score", "rmse", "mae"]:
        m[f"d_{k}"] = m[f"{k}_smooth"] - m[f"{k}_base"]

    per_seed = m.groupby("seed")[["d_score", "d_rmse", "d_mae"]].mean()
    sem = per_seed.std(ddof=1) / np.sqrt(len(per_seed))
    print("\nper-task mean (test):")
    print(m.groupby("task")[["score_base", "score_smooth", "d_score", "d_rmse", "d_mae"]].mean().round(4))
    print("\npaired over seeds (six-task mean):")
    for k in ["d_score", "d_rmse", "d_mae"]:
        mean = per_seed[k].mean()
        star = "★" if abs(mean) > 2 * sem[k] else ""
        print(f"  {k}: {mean:+.4f} (SEM {sem[k]:.4f}){star}")


if __name__ == "__main__":
    main()
