import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DDQN results across multiple run directories.")
    parser.add_argument("run_dirs", nargs="+", help="One or more ddqn sweep run directories.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where combined plots will be written.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=25,
        help="Rolling window for smoothed training curves.",
    )
    parser.add_argument(
        "--step-bin",
        type=int,
        default=0,
        help="If >0 and plotting by steps, aggregate training points into this many-step bins.",
    )
    return parser.parse_args()


def load_concat(run_dirs, filename):
    frames = []
    for run_dir in run_dirs:
        path = Path(run_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def plot_training(histories: pd.DataFrame, output_dir: Path, rolling_window: int, step_bin: int):
    import matplotlib.pyplot as plt

    for domain, domain_df in histories.groupby("domain"):
        fig, ax = plt.subplots(figsize=(10, 6))
        x_label = "Episode"
        for optimizer, opt_df in domain_df.groupby("optimizer"):
            x_col = "episode"
            if "total_steps" in opt_df.columns and opt_df["total_steps"].notna().any():
                x_col = "total_steps"
                x_label = "Environment steps"
            curve = (
                opt_df.groupby(x_col, as_index=False)["episode_reward"]
                .mean()
                .sort_values(x_col)
            )
            if x_col == "total_steps" and step_bin > 0:
                curve = curve.assign(total_steps=(curve["total_steps"] // step_bin) * step_bin)
                curve = (
                    curve.groupby("total_steps", as_index=False)["episode_reward"]
                    .mean()
                    .sort_values("total_steps")
                )
            curve["smoothed_reward"] = curve["episode_reward"].rolling(
                window=min(rolling_window, len(curve)),
                min_periods=1,
            ).mean()
            ax.plot(curve[x_col], curve["smoothed_reward"], label=optimizer)

        ax.set_title(f"{domain.title()} Training Reward")
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"Mean reward ({rolling_window}-episode rolling)")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{domain}_training_reward_combined.png", dpi=150)
        plt.close(fig)


def plot_baseline_evals(evals: pd.DataFrame, histories: pd.DataFrame, output_dir: Path):
    import matplotlib.pyplot as plt

    evals_with_steps = evals.copy()
    if ("total_steps" not in evals_with_steps.columns) or (not evals_with_steps["total_steps"].notna().any()):
        if "total_steps" in histories.columns:
            step_lookup = histories[
                ["domain", "optimizer", "seed", "episode", "total_steps"]
            ].drop_duplicates(subset=["domain", "optimizer", "seed", "episode"], keep="last")
            evals_with_steps = evals_with_steps.merge(
                step_lookup,
                on=["domain", "optimizer", "seed", "episode"],
                how="left",
            )

    for domain, domain_df in evals.groupby("domain"):
        fig, ax = plt.subplots(figsize=(10, 6))
        x_label = "Episode"
        for optimizer, opt_df in domain_df.groupby("optimizer"):
            x_col = "episode"
            if "total_steps" in evals_with_steps.columns:
                opt_steps = evals_with_steps[
                    (evals_with_steps["domain"] == domain) & (evals_with_steps["optimizer"] == optimizer)
                ]
                if not opt_steps.empty and opt_steps["total_steps"].notna().any():
                    opt_df = opt_steps
                    x_col = "total_steps"
                    x_label = "Environment steps"
            curve = (
                opt_df.groupby(x_col, as_index=False)["eval_mean_reward"]
                .mean()
                .sort_values(x_col)
            )
            ax.plot(curve[x_col], curve["eval_mean_reward"], marker="o", label=optimizer)

        ax.set_title(f"{domain.title()} Baseline Evaluation Reward")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Mean evaluation reward")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{domain}_baseline_eval_reward_combined.png", dpi=150)
        plt.close(fig)


def plot_shifted_evals(shift_evals: pd.DataFrame, output_dir: Path):
    import matplotlib.pyplot as plt

    for domain, domain_df in shift_evals.groupby("domain"):
        for shift_family, shift_df in domain_df.groupby("shift_family"):
            pivot = (
                shift_df.groupby(["shift_value", "optimizer"], as_index=False)["mean_reward"]
                .mean()
                .pivot(index="shift_value", columns="optimizer", values="mean_reward")
                .sort_index()
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"{domain.title()} Shifted Reward: {shift_family}")
            ax.set_xlabel("Shift setting")
            ax.set_ylabel("Mean reward")
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            fig.savefig(output_dir / f"{domain}_{shift_family}_shift_reward_combined.png", dpi=150)
            plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    histories = load_concat(args.run_dirs, "all_histories.csv")
    evals = load_concat(args.run_dirs, "all_evals.csv")
    shift_evals = load_concat(args.run_dirs, "all_shift_evals.csv")

    plot_training(histories, output_dir, args.rolling_window, args.step_bin)
    plot_baseline_evals(evals, histories, output_dir)
    plot_shifted_evals(shift_evals, output_dir)

    print(f"Combined plots written to: {output_dir}")


if __name__ == "__main__":
    main()
