import argparse
import re
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot AdamW update-term diagnostics over time.")
    parser.add_argument("run_dir", help="Path to a ddqn_nonstationary_sweep run directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots (defaults to <run_dir>/plots/adamw_update_terms).",
    )
    parser.add_argument("--domains", nargs="+", default=None, help="Optional domain filter.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Optional seed filter.")
    parser.add_argument("--param-name-contains", default=None, help="Optional parameter-name substring filter.")
    return parser.parse_args()


def sanitize_filename(value: str):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    diagnostics_dir = run_dir / "diagnostics"
    paths = sorted(diagnostics_dir.glob("*_adamw_momentum.csv"))
    if not paths:
        raise FileNotFoundError(f"No AdamW diagnostics found in: {diagnostics_dir}")

    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    df = df[df["optimizer"] == "adamw"].copy()
    required = {"m_hat_l2", "sqrt_v_hat_l2", "parenthesized_term_l2"}
    if not required.issubset(set(df.columns)):
        missing = sorted(required - set(df.columns))
        raise ValueError(
            "AdamW logs are missing required columns "
            f"{missing}. Rerun training with the updated logger."
        )

    if args.domains:
        df = df[df["domain"].isin(args.domains)]
    if args.seeds:
        df = df[df["seed"].isin(args.seeds)]
    if args.param_name_contains:
        df = df[df["param_name"].str.contains(args.param_name_contains, regex=False)]
    if df.empty:
        raise ValueError("No rows matched selected filters.")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "plots" / "adamw_update_terms"
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("*.png"):
        existing.unlink()

    import matplotlib.pyplot as plt

    plot_count = 0
    grouped = df.groupby(["domain", "param_name"], as_index=False)
    for (domain, param_name), group in grouped:
        curve = (
            group.groupby("optimizer_step", as_index=False)
            .agg(
                m_hat_l2=("m_hat_l2", "mean"),
                sqrt_v_hat_l2=("sqrt_v_hat_l2", "mean"),
                parenthesized_term_l2=("parenthesized_term_l2", "mean"),
            )
            .sort_values("optimizer_step")
        )
        if curve.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(curve["optimizer_step"], curve["m_hat_l2"], label="||m_hat||_2", linewidth=1.8)
        ax.plot(curve["optimizer_step"], curve["sqrt_v_hat_l2"], label="||sqrt(v_hat)||_2", linewidth=1.8)
        ax.plot(curve["optimizer_step"], curve["parenthesized_term_l2"], label="||m_hat/(sqrt(v_hat)+eps)+wd*theta||_2", linewidth=1.8)
        n_seeds = int(group["seed"].nunique())
        ax.set_title(f"{domain} param={param_name} AdamW terms (seed-avg, n={n_seeds})")
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("L2 magnitude")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        out_name = f"{domain}_{sanitize_filename(param_name)}_adamw_update_terms_seed_avg.png"
        fig.savefig(output_dir / out_name, dpi=150)
        plt.close(fig)
        plot_count += 1

    print(f"Wrote {plot_count} AdamW update-term plots to: {output_dir}")


if __name__ == "__main__":
    main()
