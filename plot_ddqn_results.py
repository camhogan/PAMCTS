import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DDQN sweep training and evaluation results.")
    parser.add_argument("run_dir", help="Path to a ddqn_nonstationary_sweep run directory.")
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=25,
        help="Rolling window for smoothed training curves.",
    )
    return parser.parse_args()


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def plot_training(histories: pd.DataFrame, plots_dir: Path, rolling_window: int):
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
        fig.savefig(plots_dir / f"{domain}_training_reward.png", dpi=150)
        plt.close(fig)


def plot_baseline_evals(evals: pd.DataFrame, plots_dir: Path):
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
        fig.savefig(plots_dir / f"{domain}_baseline_eval_reward.png", dpi=150)
        plt.close(fig)


def plot_shifted_evals(shift_evals: pd.DataFrame, plots_dir: Path):
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
            fig.savefig(plots_dir / f"{domain}_{shift_family}_shift_reward.png", dpi=150)
            plt.close(fig)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    histories_path = run_dir / "all_histories.csv"
    evals_path = run_dir / "all_evals.csv"
    shift_path = run_dir / "all_shift_evals.csv"

    require_file(histories_path)
    require_file(evals_path)
    require_file(shift_path)

    histories = pd.read_csv(histories_path)
    evals = pd.read_csv(evals_path)
    shift_evals = pd.read_csv(shift_path)

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_training(histories, plots_dir, args.rolling_window)
    plot_baseline_evals(evals, plots_dir)
    plot_shifted_evals(shift_evals, plots_dir)

    print(f"Plots written to: {plots_dir}")


if __name__ == "__main__":
    main()
