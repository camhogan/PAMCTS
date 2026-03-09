import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Muon update singular-value trajectories from diagnostics logs.")
    parser.add_argument("run_dir", help="Path to a ddqn_nonstationary_sweep run directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where Muon spectrum plots will be written (defaults to <run_dir>/plots/muon_spectrum).",
    )
    parser.add_argument(
        "--max-singular-indices",
        type=int,
        default=8,
        help="Maximum singular indices to plot per parameter (0 plots all available indices).",
    )
    parser.add_argument("--domains", nargs="+", default=None, help="Optional domain filters.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Optional seed filters.")
    parser.add_argument(
        "--param-name-contains",
        default=None,
        help="Optional substring filter for parameter name.",
    )
    parser.add_argument(
        "--condition-eps",
        type=float,
        default=1e-12,
        help="Epsilon floor used for clipped condition number sigma_max/max(sigma_min, eps).",
    )
    return parser.parse_args()


def sanitize_filename(value: str):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def load_spectrum_logs(run_dir: Path):
    diagnostics_dir = run_dir / "diagnostics"
    if not diagnostics_dir.exists():
        raise FileNotFoundError(f"Diagnostics directory not found: {diagnostics_dir}")

    paths = sorted(diagnostics_dir.glob("*_muon_update_spectrum.csv"))
    if not paths:
        raise FileNotFoundError(f"No Muon spectrum CSV files found in: {diagnostics_dir}")

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


def plot_param_spectrum(param_df: pd.DataFrame, output_path: Path, max_singular_indices: int):
    import matplotlib.pyplot as plt

    indices = sorted(param_df["singular_index"].unique().tolist())
    if max_singular_indices > 0:
        indices = indices[:max_singular_indices]

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.viridis([i / max(1, len(indices) - 1) for i in range(len(indices))])
    style = {"raw_update": "--", "ortho_update": "-"}
    kind_label = {"raw_update": "raw", "ortho_update": "ortho"}

    for idx_pos, singular_idx in enumerate(indices):
        color = colors[idx_pos]
        idx_df = param_df[param_df["singular_index"] == singular_idx]
        for update_kind in ("raw_update", "ortho_update"):
            curve = (
                idx_df[idx_df["update_kind"] == update_kind]
                .groupby("optimizer_step", as_index=False)["singular_value"]
                .mean()
                .sort_values("optimizer_step")
            )
            if curve.empty:
                continue
            ax.plot(
                curve["optimizer_step"],
                curve["singular_value"],
                linestyle=style[update_kind],
                color=color,
                linewidth=1.4,
                alpha=0.95,
                label=f"s{singular_idx} {kind_label[update_kind]}",
            )

    domain = str(param_df["domain"].iloc[0])
    param_name = str(param_df["param_name"].iloc[0])
    num_seeds = int(param_df["seed"].nunique()) if "seed" in param_df.columns else 1
    ax.set_title(f"{domain} param={param_name} (seed-avg, n={num_seeds})")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Singular value")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_param_condition_number(param_df: pd.DataFrame, output_path: Path, condition_eps: float):
    import matplotlib.pyplot as plt

    if "condition_number" not in param_df.columns:
        return

    fig, ax = plt.subplots(figsize=(11, 4.5))
    style = {"raw_update": "--", "ortho_update": "-"}
    color = {"raw_update": "#1f77b4", "ortho_update": "#d62728"}
    label = {"raw_update": "raw update", "ortho_update": "orthogonalized update"}

    any_curve = False
    for update_kind in ("raw_update", "ortho_update"):
        kind_df = param_df[param_df["update_kind"] == update_kind].copy()
        if kind_df.empty:
            continue
        kind_df["condition_number"] = pd.to_numeric(kind_df["condition_number"], errors="coerce")
        kind_df["finite_condition_number"] = kind_df["condition_number"].where(np.isfinite(kind_df["condition_number"]), np.nan)

        # Use clipped condition number so rank-deficient updates remain visible on plots.
        if "sigma_max" in kind_df.columns and "sigma_min" in kind_df.columns:
            sigma_max = pd.to_numeric(kind_df["sigma_max"], errors="coerce")
            sigma_min = pd.to_numeric(kind_df["sigma_min"], errors="coerce").clip(lower=condition_eps)
            kind_df["clipped_condition_number"] = sigma_max / sigma_min
        else:
            kind_df["clipped_condition_number"] = kind_df["finite_condition_number"]

        curve = (
            kind_df.groupby("optimizer_step", as_index=False)
            .agg(
                clipped_condition_number=("clipped_condition_number", "median"),
                finite_condition_number=("finite_condition_number", "median"),
            )
            .sort_values("optimizer_step")
        )
        if curve["clipped_condition_number"].notna().sum() == 0:
            continue

        ax.plot(
            curve["optimizer_step"],
            curve["clipped_condition_number"],
            linestyle=style[update_kind],
            color=color[update_kind],
            linewidth=1.8,
            label=f"{label[update_kind]} (clipped)",
        )
        if curve["finite_condition_number"].notna().sum() > 0:
            ax.plot(
                curve["optimizer_step"],
                curve["finite_condition_number"],
                linestyle=":",
                color=color[update_kind],
                linewidth=1.2,
                alpha=0.8,
                label=f"{label[update_kind]} (finite-only)",
            )
        any_curve = True

    domain = str(param_df["domain"].iloc[0])
    param_name = str(param_df["param_name"].iloc[0])
    num_seeds = int(param_df["seed"].nunique()) if "seed" in param_df.columns else 1
    ax.set_title(f"{domain} param={param_name} condition number (seed-avg, n={num_seeds})")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Condition number (log scale)")
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    if any_curve:
        ax.legend(fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "No finite/clipped condition number values available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "plots" / "muon_spectrum"
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("*.png"):
        existing.unlink()

    all_data = load_spectrum_logs(run_dir)
    all_data = maybe_filter(all_data, args)
    if all_data.empty:
        raise ValueError("No Muon spectrum rows match the selected filters.")

    plot_count = 0
    grouped = all_data.groupby(["domain", "param_name"], as_index=False)
    for (domain, param_name), param_df in grouped:
        safe_param = sanitize_filename(param_name)
        spectrum_filename = f"{domain}_{safe_param}_update_spectrum_seed_avg.png"
        cond_filename = f"{domain}_{safe_param}_condition_number_seed_avg.png"
        plot_param_spectrum(param_df, output_dir / spectrum_filename, args.max_singular_indices)
        plot_param_condition_number(param_df, output_dir / cond_filename, args.condition_eps)
        plot_count += 1

    print(f"Wrote {plot_count} Muon spectrum plots to: {output_dir}")


if __name__ == "__main__":
    main()
