"""Build an additive ablation ladder from a full factorial table.

Ordering rule (declare it before looking at test results): start from the
baseline and, at each step, add the component with the largest marginal gain
(forward stepwise). Also prints average main effects and the leave-one-out
table so the chosen order can be shown not to be cherry-picked.

Input CSV columns: one 0/1 column per component, plus `score`
(six-task mean Score). Extra metric columns (rmse, mae) are carried along.

Usage:
  python tools/factorial_ablation.py docs/factorial_current.csv --components FPT EDL SA
"""
import argparse
import itertools

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--components", nargs="+", required=True)
    ap.add_argument("--metric", default="score")
    ap.add_argument("--lower-is-better", action="store_true")
    args = ap.parse_args()
    comps, metric = args.components, args.metric
    sign = -1 if args.lower_is_better else 1

    df = pd.read_csv(args.csv)
    table = {tuple(int(r[c]) for c in comps): r for _, r in df.iterrows()}

    def val(active):
        key = tuple(int(c in active) for c in comps)
        return table[key][metric] if key in table else None

    missing = [
        "+".join(c for c, on in zip(comps, key) if on) or "baseline"
        for key in itertools.product([0, 1], repeat=len(comps)) if key not in table
    ]
    if missing:
        print("missing cells (run these first):", ", ".join(missing))

    print("\nforward stepwise ladder:")
    active, cur = set(), val(set())
    print(f"  S0 baseline            {cur:.4f}")
    step = 1
    while len(active) < len(comps):
        gains = {c: val(active | {c}) for c in comps if c not in active}
        known = {c: v for c, v in gains.items() if v is not None}
        if not known:
            print("  (stopped: next cells missing)")
            break
        c = max(known, key=lambda k: sign * known[k])
        delta = known[c] - cur
        flag = "" if sign * delta > 0 else "  <-- not monotonic"
        print(f"  S{step} +{c:<20} {known[c]:.4f}  ({delta:+.4f}){flag}")
        active.add(c)
        cur = known[c]
        step += 1

    print("\naverage main effect (mean over all contexts where both cells exist):")
    for c in comps:
        diffs = []
        for rest in itertools.product([0, 1], repeat=len(comps) - 1):
            others = {o for o, on in zip([x for x in comps if x != c], rest) if on}
            a, b = val(others), val(others | {c})
            if a is not None and b is not None:
                diffs.append(b - a)
        if diffs:
            print(f"  {c:<6} {sum(diffs) / len(diffs):+.4f}  over {len(diffs)} contexts: "
                  + ", ".join(f"{d:+.4f}" for d in diffs))

    full = val(set(comps))
    if full is not None:
        print(f"\nleave-one-out from full model ({full:.4f}):")
        for c in comps:
            v = val(set(comps) - {c})
            print(f"  w/o {c:<6} " + (f"{v:.4f}  ({v - full:+.4f})" if v is not None else "missing"))


if __name__ == "__main__":
    main()
