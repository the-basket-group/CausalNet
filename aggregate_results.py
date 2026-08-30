"""Aggregate the RQ1 sweep into one side-by-side table.

Scans ./results/<mode>_seed<seed>/ folders produced by run_all.sh, pools each run's
per-subject predictions (the *_acc.txt files), and reports ACC / WAR / UAR / WF1 / UF1
per variant, averaged across seeds (mean +/- std). Reads the same prediction files that
calculate_all_results.py uses, so the numbers match.

    python aggregate_results.py                          # all modes, seeds found on disk
    python aggregate_results.py --modes shared disentangled --seeds 2025
"""
import os
import ast
import glob
import argparse
import numpy as np
from sklearn.metrics import recall_score, f1_score, accuracy_score

LABELS = [0, 1, 2]


def load_run(run_dir):
    """Pool [pred, truth] pairs from every *_acc.txt in a run folder."""
    pairs = []
    for txt in glob.glob(os.path.join(run_dir, "*_acc.txt")):
        with open(txt) as f:
            lines = f.readlines()
        # Line 3 is 'matrix_acc: [[pred, truth], ...]' (same layout calculate_all_results reads).
        matrix_line = next((ln for ln in lines if ln.startswith("matrix_acc:")), None)
        if matrix_line is None:
            continue
        pairs.extend(ast.literal_eval(matrix_line.split("matrix_acc:", 1)[1].strip()))
    return pairs


def metrics(pairs):
    if not pairs:
        return None
    y_pred = [p for p, _ in pairs]
    y_true = [t for _, t in pairs]
    return dict(
        acc=accuracy_score(y_true, y_pred),
        war=recall_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        uar=recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        wf1=f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        uf1=f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
    )


def discover_seeds(results_dir, modes):
    seeds = set()
    for m in modes:
        for d in glob.glob(os.path.join(results_dir, m + "_seed*")):
            tail = os.path.basename(d).split("_seed")[-1]
            if tail.isdigit():
                seeds.add(int(tail))
    return sorted(seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(".", "results"))
    ap.add_argument("--modes", nargs="+", default=["shared", "disentangled", "shared_wide"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Seeds to include; default = whatever is found on disk.")
    args = ap.parse_args()

    seeds = args.seeds or discover_seeds(args.results_dir, args.modes)
    cols = ["acc", "war", "uar", "wf1", "uf1"]

    print("Combined 3-class metrics (pooled over all LOSO subjects)")
    print("seeds:", seeds or "(none found)")
    print()
    header = f"{'variant':14s} {'seed':>6s} " + " ".join(f"{c.upper():>7s}" for c in cols)
    print(header)
    print("-" * len(header))

    for mode in args.modes:
        per_seed = {c: [] for c in cols}
        for s in seeds:
            run_dir = os.path.join(args.results_dir, f"{mode}_seed{s}")
            mt = metrics(load_run(run_dir))
            if mt is None:
                print(f"{mode:14s} {s:>6d}   (no results yet)")
                continue
            for c in cols:
                per_seed[c].append(mt[c])
            print(f"{mode:14s} {s:>6d} " + " ".join(f"{mt[c]:7.4f}" for c in cols))
        n = len(per_seed["acc"])
        if n:
            mean = " ".join(f"{np.mean(per_seed[c]):7.4f}" for c in cols)
            print(f"{mode:14s} {'mean':>6s} " + mean)
            if n > 1:
                std = " ".join(f"{np.std(per_seed[c]):7.4f}" for c in cols)
                print(f"{mode:14s} {'std':>6s} " + std)
        print()


if __name__ == "__main__":
    main()