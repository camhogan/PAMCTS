import argparse
import json
import os
import traceback
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from pauct.cartpole.MCTS.PAMCTS_Outside import PAMCTS_Outside
from pauct.cartpole.MCTS.ddqn_agent import DDQN_Learning_Agent

try:
    from gym_compat import make_legacy_env
except ModuleNotFoundError:
    from Cartpole.gym_compat import make_legacy_env


DEFAULT_WEIGHTS = Path("Cartpole/Network_Files/duel_dqn_CartPole-v1_weights_2500.h5f")


def parse_args():
    parser = argparse.ArgumentParser(description="Run CartPole PA-MCTS evaluations and save CSV summaries.")
    parser.add_argument(
        "--weights-path",
        default=str(DEFAULT_WEIGHTS),
        help="Path to the trained DDQN weights.",
    )
    parser.add_argument(
        "--output-dir",
        default="Cartpole/results/pamcts",
        help="Directory where run artifacts will be written.",
    )
    parser.add_argument(
        "--gravities",
        nargs="+",
        type=float,
        default=[9.8, 20.0, 50.0, 500.0],
        help="Gravity settings to evaluate.",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Alpha values to evaluate.",
    )
    parser.add_argument(
        "--simulations",
        nargs="+",
        type=int,
        default=[25, 50, 100, 200, 300, 400, 500, 1000],
        help="Number of MCTS simulations per decision.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=60,
        help="How many runs to execute per parameter combination.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Worker process count. Defaults to a laptop-safe value.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=2500,
        help="Episode length cap.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.999,
        help="Discount factor for PA-MCTS.",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=50.0,
        help="Exploration constant for PA-MCTS.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=2,
        help="Base seed used to derive per-sample environment seeds.",
    )
    return parser.parse_args()


def build_tasks(args, weights_path):
    tasks = []
    for gravity in args.gravities:
        for alpha in args.alphas:
            for simulations in args.simulations:
                for sample_id in range(args.sample_count):
                    tasks.append(
                        {
                            "gravity": gravity,
                            "alpha": alpha,
                            "simulations": simulations,
                            "sample_id": sample_id,
                            "max_episode_steps": args.max_episode_steps,
                            "gamma": args.gamma,
                            "c_puct": args.c_puct,
                            "seed": args.seed_base + sample_id,
                            "weights_path": str(weights_path),
                        }
                    )
    return tasks


def weights_path_available(weights_path):
    if weights_path.exists():
        return True
    index_path = Path(f"{weights_path}.index")
    return index_path.exists()


def evaluate_task(task):
    result = {
        "gravity": task["gravity"],
        "alpha": task["alpha"],
        "simulations": task["simulations"],
        "sample_id": task["sample_id"],
    }

    try:
        env = make_legacy_env("CartPole-v1")
        env.env.gravity = task["gravity"]
        env._max_episode_steps = task["max_episode_steps"]
        current_state = env.reset(seed=task["seed"])

        nb_actions = env.action_space.n
        learning_agent = DDQN_Learning_Agent(
            number_of_actions=nb_actions,
            env_obs_space_shape=env.observation_space.shape,
        )
        learning_agent.load_saved_weights(task["weights_path"])

        search_agent = PAMCTS_Outside(
            gamma=task["gamma"],
            learning_agent=learning_agent,
            alpha=task["alpha"],
            num_iter=task["simulations"],
            max_depth=500,
            verbose=False,
            c_puct=task["c_puct"],
        )

        cumulative_reward = 0.0
        step_counter = 0
        terminated = False

        while not terminated and step_counter < task["max_episode_steps"]:
            action = search_agent.get_action(current_state, env)
            current_state, reward, terminated, _info = env.step(action)
            cumulative_reward += reward
            step_counter += 1

        result.update(
            {
                "cumulative_reward": cumulative_reward,
                "step_counter": step_counter,
                "success": True,
                "error": "",
            }
        )
    except Exception:
        result.update(
            {
                "cumulative_reward": None,
                "step_counter": None,
                "success": False,
                "error": traceback.format_exc(),
            }
        )

    return result


def write_manifest(run_dir, args, task_count, weights_path):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "weights_path": str(weights_path),
        "task_count": task_count,
        "processes": args.processes,
        "gravities": args.gravities,
        "alphas": args.alphas,
        "simulations": args.simulations,
        "sample_count": args.sample_count,
        "max_episode_steps": args.max_episode_steps,
        "gamma": args.gamma,
        "c_puct": args.c_puct,
        "seed_base": args.seed_base,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def append_result(csv_path, row):
    frame = pd.DataFrame([row])
    header = not csv_path.exists()
    frame.to_csv(csv_path, mode="a", header=header, index=False)


def write_summary(results_csv, summary_csv):
    results = pd.read_csv(results_csv)
    completed = results[results["success"] == True].copy()
    if completed.empty:
        pd.DataFrame(
            columns=[
                "gravity",
                "alpha",
                "simulations",
                "num_runs",
                "mean_reward",
                "std_reward",
                "mean_steps",
                "std_steps",
            ]
        ).to_csv(summary_csv, index=False)
        return

    summary = (
        completed.groupby(["gravity", "alpha", "simulations"], as_index=False)
        .agg(
            num_runs=("cumulative_reward", "count"),
            mean_reward=("cumulative_reward", "mean"),
            std_reward=("cumulative_reward", "std"),
            mean_steps=("step_counter", "mean"),
            std_steps=("step_counter", "std"),
        )
        .sort_values(["gravity", "alpha", "simulations"])
    )
    summary.to_csv(summary_csv, index=False)


def main():
    args = parse_args()
    weights_path = Path(args.weights_path)
    if not weights_path_available(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results_csv = run_dir / "raw_results.csv"
    summary_csv = run_dir / "summary.csv"
    failures_csv = run_dir / "failures.csv"

    tasks = build_tasks(args, weights_path)
    write_manifest(run_dir, args, len(tasks), weights_path)

    failures = []
    with Pool(processes=args.processes) as pool:
        for row in pool.imap_unordered(evaluate_task, tasks):
            append_result(results_csv, row)
            if not row["success"]:
                failures.append(row)

    write_summary(results_csv, summary_csv)

    if failures:
        pd.DataFrame(failures).to_csv(failures_csv, index=False)

    print(f"Run directory: {run_dir}")
    print(f"Raw results: {results_csv}")
    print(f"Summary: {summary_csv}")
    if failures:
        print(f"Failures: {failures_csv}")
    print("experiments completed")


if __name__ == "__main__":
    main()
