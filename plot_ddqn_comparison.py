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
    return parser.parse_args()


def load_concat(run_dirs, filename):
    frames = []
    for run_dir in run_dirs:
        path = Path(run_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def plot_training(histories: pd.DataFrame, output_dir: Path, rolling_window: int):
    import matplotlib.pyplot as plt

    for domain, domain_df in histories.groupby("domain"):
        fig, ax = plt.subplots(figsize=(10, 6))
        for optimizer, opt_df in domain_df.groupby("optimizer"):
            curve = (
                opt_df.groupby("episode", as_index=False)["episode_reward"]
                .mean()
                .sort_values("episode")
            )
            curve["smoothed_reward"] = curve["episode_reward"].rolling(
                window=min(rolling_window, len(curve)),
                min_periods=1,
            ).mean()
            ax.plot(curve["episode"], curve["smoothed_reward"], label=optimizer)

        ax.set_title(f"{domain.title()} Training Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel(f"Mean reward ({rolling_window}-episode rolling)")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{domain}_training_reward_combined.png", dpi=150)
        plt.close(fig)


def plot_baseline_evals(evals: pd.DataFrame, output_dir: Path):
    import matplotlib.pyplot as plt

    for domain, domain_df in evals.groupby("domain"):
        fig, ax = plt.subplots(figsize=(10, 6))
        for optimizer, opt_df in domain_df.groupby("optimizer"):
            curve = (
                opt_df.groupby("episode", as_index=False)["eval_mean_reward"]
                .mean()
                .sort_values("episode")
            )
            ax.plot(curve["episode"], curve["eval_mean_reward"], marker="o", label=optimizer)

        ax.set_title(f"{domain.title()} Baseline Evaluation Reward")
        ax.set_xlabel("Episode")
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

    plot_training(histories, output_dir, args.rolling_window)
    plot_baseline_evals(evals, output_dir)
    plot_shifted_evals(shift_evals, output_dir)

    print(f"Combined plots written to: {output_dir}")


if __name__ == "__main__":
    main()
