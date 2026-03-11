import argparse
import re
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot momentum trajectories for AdamW vs SGDM vs NAG.")
    parser.add_argument("run_dir", help="Path to a ddqn_nonstationary_sweep run directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots (defaults to <run_dir>/plots/momentum_comparison).",
    )
    parser.add_argument("--domains", nargs="+", default=None, help="Optional domain filter.")
    parser.add_argument("--param-name-contains", default=None, help="Optional parameter-name substring filter.")
    parser.add_argument(
        "--metric",
        default="rms",
        choices=["rms", "l2", "mean_abs", "max_abs"],
        help="Momentum metric to plot.",
    )
    return parser.parse_args()


def sanitize_filename(value: str):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def load_adamw(run_dir: Path, metric: str):
    paths = sorted((run_dir / "diagnostics").glob("*_adamw_momentum.csv"))
    if not paths:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    df = df[df["optimizer"] == "adamw"].copy()
    if df.empty:
        return df
    metric_col = {
        "rms": "exp_avg_rms",
        "l2": "exp_avg_l2",
        "mean_abs": "exp_avg_mean_abs",
        "max_abs": "exp_avg_max_abs",
    }[metric]
    df["momentum_value"] = df[metric_col]
    return df[["domain", "optimizer", "seed", "optimizer_step", "param_name", "momentum_value"]]


def load_sgd(run_dir: Path):
    paths = sorted((run_dir / "diagnostics").glob("*_sgd_momentum.csv"))
    if not paths:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    metric_map = {
        "rms": "momentum_buffer_rms",
        "l2": "momentum_buffer_l2",
        "mean_abs": "momentum_buffer_mean_abs",
        "max_abs": "momentum_buffer_max_abs",
    }
    return df, metric_map


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "plots" / "momentum_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("*.png"):
        existing.unlink()

    adamw_df = load_adamw(run_dir, args.metric)
    sgd_raw = sorted((run_dir / "diagnostics").glob("*_sgd_momentum.csv"))
    if not sgd_raw:
        raise FileNotFoundError(
            f"No SGD momentum logs found in {run_dir / 'diagnostics'}. "
            "Rerun with --sgd-momentum-log-every > 0 for sgd_momentum and sgd_nag."
        )
    sgd_df = pd.concat([pd.read_csv(path) for path in sgd_raw], ignore_index=True)
    metric_col = {
        "rms": "momentum_buffer_rms",
        "l2": "momentum_buffer_l2",
        "mean_abs": "momentum_buffer_mean_abs",
        "max_abs": "momentum_buffer_max_abs",
    }[args.metric]
    sgd_df["momentum_value"] = sgd_df[metric_col]
    sgd_df = sgd_df[["domain", "optimizer", "seed", "optimizer_step", "param_name", "momentum_value"]]

    all_df = pd.concat([adamw_df, sgd_df], ignore_index=True)
    all_df = all_df[all_df["optimizer"].isin(["adamw", "sgd_momentum", "sgd_nag"])].copy()
    if args.domains:
        all_df = all_df[all_df["domain"].isin(args.domains)]
    if args.param_name_contains:
        all_df = all_df[all_df["param_name"].str.contains(args.param_name_contains, regex=False)]
    if all_df.empty:
        raise ValueError("No momentum rows matched filters.")

    import matplotlib.pyplot as plt

    plot_count = 0
    for (domain, param_name), group in all_df.groupby(["domain", "param_name"], as_index=False):
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for optimizer in ["adamw", "sgd_momentum", "sgd_nag"]:
            opt_df = group[group["optimizer"] == optimizer]
            if opt_df.empty:
                continue
            curve = (
                opt_df.groupby("optimizer_step", as_index=False)["momentum_value"]
                .mean()
                .sort_values("optimizer_step")
            )
            if curve.empty:
                continue
            n_seeds = opt_df["seed"].nunique()
            ax.plot(curve["optimizer_step"], curve["momentum_value"], linewidth=1.8, label=f"{optimizer} (n={n_seeds})")

        ax.set_title(f"{domain} param={param_name} momentum comparison ({args.metric}, seed-avg)")
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel(f"Momentum {args.metric}")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        filename = f"{domain}_{sanitize_filename(param_name)}_momentum_{args.metric}_comparison.png"
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)
        plot_count += 1

    print(f"Wrote {plot_count} momentum-comparison plots to: {output_dir}")


if __name__ == "__main__":
    main()
