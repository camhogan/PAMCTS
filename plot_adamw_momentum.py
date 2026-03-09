import argparse
import re
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot AdamW momentum (exp_avg) trajectories from diagnostics logs.")
    parser.add_argument("run_dir", help="Path to a ddqn_nonstationary_sweep run directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where AdamW momentum plots will be written (defaults to <run_dir>/plots/adamw_momentum).",
    )
    parser.add_argument("--domains", nargs="+", default=None, help="Optional domain filters.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Optional seed filters.")
    parser.add_argument(
        "--param-name-contains",
        default=None,
        help="Optional substring filter for parameter name.",
    )
    return parser.parse_args()


def sanitize_filename(value: str):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def load_momentum_logs(run_dir: Path):
    diagnostics_dir = run_dir / "diagnostics"
    if not diagnostics_dir.exists():
        raise FileNotFoundError(f"Diagnostics directory not found: {diagnostics_dir}")

    paths = sorted(diagnostics_dir.glob("*_adamw_momentum.csv"))
    if not paths:
        raise FileNotFoundError(f"No AdamW momentum CSV files found in: {diagnostics_dir}")

    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def maybe_filter(df: pd.DataFrame, args):
    if args.domains:
        df = df[df["domain"].isin(args.domains)]
    if args.seeds:
        df = df[df["seed"].isin(args.seeds)]
    if args.param_name_contains:
        df = df[df["param_name"].str.contains(args.param_name_contains, regex=False)]
    return df


def plot_param_momentum(param_df: pd.DataFrame, output_path: Path):
    import matplotlib.pyplot as plt

    curve = (
        param_df.groupby("optimizer_step", as_index=False)
        .agg(
            exp_avg_l2=("exp_avg_l2", "mean"),
            exp_avg_rms=("exp_avg_rms", "mean"),
            exp_avg_mean_abs=("exp_avg_mean_abs", "mean"),
            exp_avg_max_abs=("exp_avg_max_abs", "mean"),
            exp_avg_sq_l2=("exp_avg_sq_l2", "mean"),
        )
        .sort_values("optimizer_step")
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(curve["optimizer_step"], curve["exp_avg_l2"], label="exp_avg_l2", linewidth=1.8)
    ax.plot(curve["optimizer_step"], curve["exp_avg_rms"], label="exp_avg_rms", linewidth=1.5)
    ax.plot(curve["optimizer_step"], curve["exp_avg_mean_abs"], label="exp_avg_mean_abs", linewidth=1.5)
    ax.plot(curve["optimizer_step"], curve["exp_avg_max_abs"], label="exp_avg_max_abs", linewidth=1.5)
    ax.plot(curve["optimizer_step"], curve["exp_avg_sq_l2"], label="exp_avg_sq_l2", linewidth=1.5)

    domain = str(param_df["domain"].iloc[0])
    optimizer = str(param_df["optimizer"].iloc[0])
    seed = int(param_df["seed"].iloc[0])
    param_name = str(param_df["param_name"].iloc[0])
    ax.set_title(f"{domain} {optimizer} seed={seed} param={param_name}")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Momentum magnitude")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "plots" / "adamw_momentum"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = load_momentum_logs(run_dir)
    all_data = maybe_filter(all_data, args)
    if all_data.empty:
        raise ValueError("No AdamW momentum rows match the selected filters.")

    plot_count = 0
    grouped = all_data.groupby(["domain", "optimizer", "seed", "param_name"], as_index=False)
    for (domain, optimizer, seed, param_name), param_df in grouped:
        safe_param = sanitize_filename(param_name)
        filename = f"{domain}_{optimizer}_seed{seed}_{safe_param}_adamw_momentum.png"
        plot_param_momentum(param_df, output_dir / filename)
        plot_count += 1

    print(f"Wrote {plot_count} AdamW momentum plots to: {output_dir}")


if __name__ == "__main__":
    main()
