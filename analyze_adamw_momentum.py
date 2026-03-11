import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze AdamW momentum diagnostics and correlate with eval metrics.")
    parser.add_argument("run_dir", help="Path to a ddqn_nonstationary_sweep run directory.")
    return parser.parse_args()


def load_momentum_logs(run_dir: Path):
    diagnostics_dir = run_dir / "diagnostics"
    paths = sorted(diagnostics_dir.glob("*_adamw_momentum.csv"))
    if not paths:
        raise FileNotFoundError(f"No AdamW momentum logs found in: {diagnostics_dir}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def infer_param_sizes(run_dir: Path):
    raw_results = pd.read_csv(run_dir / "raw_results.csv")
    sizes = {}
    for row in raw_results.itertuples():
        if row.optimizer != "adamw":
            continue
        key = (row.domain, row.optimizer, int(row.seed))
        if key in sizes:
            continue
        model_path = Path(row.model_path)
        if not model_path.exists():
            continue
        state = torch.load(model_path, map_location="cpu")
        sizes[key] = {name: int(tensor.numel()) for name, tensor in state.items()}
    return sizes


def compute_eval_auc(evals: pd.DataFrame):
    rows = []
    for (domain, optimizer, seed), g in evals.groupby(["domain", "optimizer", "seed"], as_index=False):
        g = g.sort_values("episode")
        x = g["episode"].to_numpy(dtype=float)
        y = g["eval_mean_reward"].to_numpy(dtype=float)
        if len(g) < 2 or x[-1] <= x[0]:
            auc = np.nan
            auc_norm = np.nan
        else:
            auc = float(np.trapz(y, x))
            auc_norm = float(auc / (x[-1] - x[0]))
        rows.append(
            {
                "domain": domain,
                "optimizer": optimizer,
                "seed": int(seed),
                "eval_auc": auc,
                "eval_auc_normalized": auc_norm,
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    momentum = load_momentum_logs(run_dir)
    evals = pd.read_csv(run_dir / "all_evals.csv")
    eval_auc = compute_eval_auc(evals)
    param_sizes = infer_param_sizes(run_dir)

    # Seed-level momentum summaries.
    seed_rows = []
    for (domain, optimizer, seed, param_name), g in momentum.groupby(
        ["domain", "optimizer", "seed", "param_name"], as_index=False
    ):
        key = (domain, optimizer, int(seed))
        numel = param_sizes.get(key, {}).get(param_name, np.nan)
        denom = np.sqrt(g["exp_avg_sq_l2"] / numel) if pd.notna(numel) else np.nan
        ratio = g["exp_avg_rms"] / denom if pd.notna(numel) else np.nan
        seed_rows.append(
            {
                "domain": domain,
                "optimizer": optimizer,
                "seed": int(seed),
                "param_name": param_name,
                "numel": numel,
                "exp_avg_rms_mean": float(g["exp_avg_rms"].mean()),
                "exp_avg_rms_std": float(g["exp_avg_rms"].std()),
                "exp_avg_mean_abs_mean": float(g["exp_avg_mean_abs"].mean()),
                "exp_avg_max_abs_mean": float(g["exp_avg_max_abs"].mean()),
                "exp_avg_sq_l2_mean": float(g["exp_avg_sq_l2"].mean()),
                "momentum_to_second_moment_ratio_mean": float(np.nanmean(ratio)) if pd.notna(numel) else np.nan,
            }
        )

    per_seed = pd.DataFrame(seed_rows)
    per_seed = per_seed.merge(
        eval_auc[["domain", "optimizer", "seed", "eval_auc_normalized"]],
        on=["domain", "optimizer", "seed"],
        how="left",
    )

    # Average across seeds (requested).
    seed_avg = (
        per_seed.groupby(["domain", "optimizer", "param_name"], as_index=False)
        .agg(
            num_seeds=("seed", "nunique"),
            exp_avg_rms_mean=("exp_avg_rms_mean", "mean"),
            exp_avg_rms_std_across_seeds=("exp_avg_rms_mean", "std"),
            exp_avg_rms_time_std_mean=("exp_avg_rms_std", "mean"),
            exp_avg_mean_abs_mean=("exp_avg_mean_abs_mean", "mean"),
            exp_avg_max_abs_mean=("exp_avg_max_abs_mean", "mean"),
            exp_avg_sq_l2_mean=("exp_avg_sq_l2_mean", "mean"),
            momentum_to_second_moment_ratio_mean=("momentum_to_second_moment_ratio_mean", "mean"),
            eval_auc_normalized_mean=("eval_auc_normalized", "mean"),
            eval_auc_normalized_std=("eval_auc_normalized", "std"),
        )
        .sort_values(["domain", "optimizer", "param_name"])
    )

    # Correlation across seeds (per domain/param) between momentum strength and eval quality.
    corr_rows = []
    for (domain, optimizer, param_name), g in per_seed.groupby(["domain", "optimizer", "param_name"], as_index=False):
        sub = g[["exp_avg_rms_mean", "momentum_to_second_moment_ratio_mean", "eval_auc_normalized"]].dropna()
        if len(sub) < 3:
            corr1 = np.nan
            corr2 = np.nan
        else:
            corr1 = float(sub["exp_avg_rms_mean"].corr(sub["eval_auc_normalized"]))
            corr2 = float(sub["momentum_to_second_moment_ratio_mean"].corr(sub["eval_auc_normalized"]))
        corr_rows.append(
            {
                "domain": domain,
                "optimizer": optimizer,
                "param_name": param_name,
                "num_seeds_for_corr": len(sub),
                "corr_exp_avg_rms_vs_eval_auc_norm": corr1,
                "corr_ratio_vs_eval_auc_norm": corr2,
            }
        )
    corr_df = pd.DataFrame(corr_rows).sort_values(["domain", "optimizer", "param_name"])

    per_seed_path = run_dir / "all_adamw_momentum_stats_by_seed.csv"
    seed_avg_path = run_dir / "summary_adamw_momentum_seed_avg.csv"
    corr_path = run_dir / "summary_adamw_momentum_correlations.csv"
    per_seed.to_csv(per_seed_path, index=False)
    seed_avg.to_csv(seed_avg_path, index=False)
    corr_df.to_csv(corr_path, index=False)

    print(f"Wrote: {per_seed_path}")
    print(f"Wrote: {seed_avg_path}")
    print(f"Wrote: {corr_path}")


if __name__ == "__main__":
    main()
