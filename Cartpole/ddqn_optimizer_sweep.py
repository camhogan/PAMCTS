import argparse
import json
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gym
import numpy as np
import pandas as pd
import torch
from torch import nn


ENV_NAME = "CartPole-v1"


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def append(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
        )

    def forward(self, x):
        return self.layers(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a DDQN optimizer sweep on CartPole.")
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["sgd", "sgd_momentum", "sgd_nag", "adamw"],
        choices=["sgd", "sgd_momentum", "sgd_nag", "adamw"],
        help="Optimizers to compare.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[2, 3, 4, 5, 6], help="Seeds to run.")
    parser.add_argument("--episodes", type=int, default=400, help="Training episodes per run.")
    parser.add_argument("--max-steps", type=int, default=500, help="Step cap per episode.")
    parser.add_argument("--batch-size", type=int, default=64, help="Replay batch size.")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer capacity.")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Replay warmup before updates.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--target-update-frequency", type=int, default=1000, help="Target sync frequency in env steps.")
    parser.add_argument("--train-frequency", type=int, default=1, help="Gradient updates every N env steps.")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Evaluation episodes per checkpoint/final run.")
    parser.add_argument(
        "--shift-eval-episodes",
        type=int,
        default=20,
        help="Evaluation episodes for each shifted environment setting.",
    )
    parser.add_argument("--eval-every", type=int, default=25, help="Evaluate every N training episodes.")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon.")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Final epsilon.")
    parser.add_argument("--epsilon-decay-steps", type=int, default=20000, help="Linear epsilon decay horizon in env steps.")
    parser.add_argument("--shared-lr", type=float, default=None, help="Override learning rate for all optimizers.")
    parser.add_argument("--sgd-lr", type=float, default=0.01, help="Learning rate for plain SGD.")
    parser.add_argument("--sgd-momentum-lr", type=float, default=0.01, help="Learning rate for SGD with momentum.")
    parser.add_argument("--sgd-nag-lr", type=float, default=0.01, help="Learning rate for SGD with Nesterov momentum.")
    parser.add_argument("--adamw-lr", type=float, default=0.001, help="Learning rate for AdamW.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum used for momentum/NAG.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--output-dir", default="Cartpole/results/optimizer_sweep", help="Sweep output directory.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument("--save-plots", action="store_true", help="Generate PNG summary plots.")
    parser.add_argument(
        "--gravity-settings",
        nargs="+",
        type=float,
        default=[9.8, 20.0, 50.0, 500.0],
        help="CartPole gravity settings for post-training environment-shift evaluation.",
    )
    parser.add_argument(
        "--masscart-settings",
        nargs="+",
        type=float,
        default=[0.1, 1.0, 1.2, 1.3, 1.5],
        help="CartPole cart-mass settings for post-training environment-shift evaluation.",
    )
    return parser.parse_args()


def make_env(seed: int, max_steps: int, gravity=None, masscart=None):
    env = gym.make(ENV_NAME)
    env._max_episode_steps = max_steps
    env.action_space.seed(seed)
    if gravity is not None:
        env.unwrapped.gravity = gravity
    if masscart is not None:
        env.unwrapped.masscart = masscart
        env.unwrapped.total_mass = env.unwrapped.masspole + env.unwrapped.masscart
        env.unwrapped.polemass_length = env.unwrapped.masspole * env.unwrapped.length
    return env


def reset_env(env, seed=None):
    state, _info = env.reset(seed=seed)
    return np.asarray(state, dtype=np.float32)


def step_env(env, action: int):
    next_state, reward, terminated, truncated, _info = env.step(action)
    done = terminated or truncated
    return np.asarray(next_state, dtype=np.float32), float(reward), done


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def optimizer_spec(name: str, args):
    if name == "sgd":
        lr = args.shared_lr if args.shared_lr is not None else args.sgd_lr
        return {"name": name, "lr": lr, "momentum": 0.0, "nesterov": False}
    if name == "sgd_momentum":
        lr = args.shared_lr if args.shared_lr is not None else args.sgd_momentum_lr
        return {"name": name, "lr": lr, "momentum": args.momentum, "nesterov": False}
    if name == "sgd_nag":
        lr = args.shared_lr if args.shared_lr is not None else args.sgd_nag_lr
        return {"name": name, "lr": lr, "momentum": args.momentum, "nesterov": True}
    if name == "adamw":
        lr = args.shared_lr if args.shared_lr is not None else args.adamw_lr
        return {"name": name, "lr": lr, "weight_decay": args.weight_decay}
    raise ValueError(f"Unsupported optimizer: {name}")


def build_optimizer(parameters, spec):
    if spec["name"] == "adamw":
        return torch.optim.AdamW(parameters, lr=spec["lr"], weight_decay=spec["weight_decay"])
    return torch.optim.SGD(
        parameters,
        lr=spec["lr"],
        momentum=spec.get("momentum", 0.0),
        nesterov=spec.get("nesterov", False),
    )


def select_action(policy_net, state, epsilon, action_dim, device):
    if random.random() < epsilon:
        return random.randrange(action_dim)
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return int(policy_net(state_tensor).argmax(dim=1).item())


def linear_epsilon(step_count, start, end, decay_steps):
    if decay_steps <= 0:
        return end
    frac = min(1.0, step_count / decay_steps)
    return start + frac * (end - start)


def optimize_model(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    if len(replay_buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    q_values = policy_net(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
        next_q_values = target_net(next_states).gather(1, next_actions).squeeze(1)
        targets = rewards + gamma * (1.0 - dones) * next_q_values

    loss = nn.functional.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def evaluate(policy_net, env, episodes, max_steps, seed, device):
    rewards = []
    for episode in range(episodes):
        state = reset_env(env, seed=seed + 10_000 + episode)
        episode_reward = 0.0
        for _step in range(max_steps):
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(policy_net(state_tensor).argmax(dim=1).item())
            state, reward, done = step_env(env, action)
            episode_reward += reward
            if done:
                break
        rewards.append(episode_reward)
    return rewards


def evaluate_shift(
    policy_net,
    max_steps,
    episodes,
    seed,
    device,
    shift_name,
    shift_value,
):
    rewards = []
    for episode in range(episodes):
        env_seed = seed + 20_000 + episode
        if shift_name == "gravity":
            env = make_env(env_seed, max_steps, gravity=shift_value)
        elif shift_name == "masscart":
            env = make_env(env_seed, max_steps, masscart=shift_value)
        else:
            raise ValueError(f"Unsupported shift: {shift_name}")

        state = reset_env(env, seed=env_seed)
        episode_reward = 0.0
        for _step in range(max_steps):
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(policy_net(state_tensor).argmax(dim=1).item())
            state, reward, done = step_env(env, action)
            episode_reward += reward
            if done:
                break
        rewards.append(episode_reward)
        env.close()
    return rewards


def maybe_save_plots(run_dir: Path, history_df: pd.DataFrame, summary_df: pd.DataFrame):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not history_df.empty:
        curve_df = (
            history_df.groupby(["optimizer", "episode"], as_index=False)["episode_reward"]
            .mean()
            .sort_values(["optimizer", "episode"])
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        for optimizer, group in curve_df.groupby("optimizer"):
            ax.plot(group["episode"], group["episode_reward"], label=optimizer)
        ax.set_title("Mean Training Reward by Optimizer")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "training_reward_curves.png", dpi=150)
        plt.close(fig)

    if not summary_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(summary_df["optimizer"], summary_df["final_eval_mean_reward_mean"], yerr=summary_df["final_eval_mean_reward_std"])
        ax.set_title("Final Evaluation Reward by Optimizer")
        ax.set_ylabel("Reward")
        fig.tight_layout()
        fig.savefig(plots_dir / "final_eval_reward_summary.png", dpi=150)
        plt.close(fig)


def run_single_experiment(optimizer_name: str, seed: int, args, run_dir: Path, device: torch.device):
    spec = optimizer_spec(optimizer_name, args)
    set_all_seeds(seed)

    env = make_env(seed, args.max_steps)
    eval_env = make_env(seed + 50_000, args.max_steps)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = build_optimizer(policy_net.parameters(), spec)
    replay_buffer = ReplayBuffer(args.buffer_size)

    history_rows = []
    eval_rows = []
    shift_rows = []
    total_steps = 0
    training_start = time.time()

    for episode in range(1, args.episodes + 1):
        state = reset_env(env, seed=seed + episode)
        episode_reward = 0.0
        episode_losses = []

        for _ in range(args.max_steps):
            epsilon = linear_epsilon(total_steps, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
            action = select_action(policy_net, state, epsilon, action_dim, device)
            next_state, reward, done = step_env(env, action)
            replay_buffer.append(Transition(state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward
            total_steps += 1

            if total_steps >= args.warmup_steps and total_steps % args.train_frequency == 0:
                loss = optimize_model(
                    policy_net,
                    target_net,
                    optimizer,
                    replay_buffer,
                    args.batch_size,
                    args.gamma,
                    device,
                )
                if loss is not None:
                    episode_losses.append(loss)

            if total_steps % args.target_update_frequency == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        mean_loss = float(np.mean(episode_losses)) if episode_losses else None
        history_rows.append(
            {
                "optimizer": optimizer_name,
                "seed": seed,
                "episode": episode,
                "episode_reward": episode_reward,
                "epsilon": linear_epsilon(total_steps, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps),
                "total_steps": total_steps,
                "mean_loss": mean_loss,
            }
        )

        if episode % args.eval_every == 0 or episode == args.episodes:
            eval_rewards = evaluate(policy_net, eval_env, args.eval_episodes, args.max_steps, seed + episode, device)
            eval_rows.append(
                {
                    "optimizer": optimizer_name,
                    "seed": seed,
                    "episode": episode,
                    "eval_mean_reward": float(np.mean(eval_rewards)),
                    "eval_std_reward": float(np.std(eval_rewards)),
                }
            )

    history_df = pd.DataFrame(history_rows)
    eval_df = pd.DataFrame(eval_rows)
    model_path = run_dir / "models" / f"{optimizer_name}_seed{seed}.pt"
    history_path = run_dir / "histories" / f"{optimizer_name}_seed{seed}.csv"
    eval_path = run_dir / "evals" / f"{optimizer_name}_seed{seed}.csv"
    shift_path = run_dir / "shift_evals" / f"{optimizer_name}_seed{seed}.csv"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    shift_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy_net.state_dict(), model_path)
    history_df.to_csv(history_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    for gravity in args.gravity_settings:
        rewards = evaluate_shift(
            policy_net,
            args.max_steps,
            args.shift_eval_episodes,
            seed + 1000,
            device,
            "gravity",
            gravity,
        )
        shift_rows.append(
            {
                "optimizer": optimizer_name,
                "seed": seed,
                "shift_type": "gravity",
                "shift_value": gravity,
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
            }
        )

    for masscart in args.masscart_settings:
        rewards = evaluate_shift(
            policy_net,
            args.max_steps,
            args.shift_eval_episodes,
            seed + 2000,
            device,
            "masscart",
            masscart,
        )
        shift_rows.append(
            {
                "optimizer": optimizer_name,
                "seed": seed,
                "shift_type": "masscart",
                "shift_value": masscart,
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
            }
        )

    shift_df = pd.DataFrame(shift_rows)
    shift_df.to_csv(shift_path, index=False)

    last_20_mean = float(history_df["episode_reward"].tail(20).mean())
    final_eval_mean = float(eval_df["eval_mean_reward"].iloc[-1]) if not eval_df.empty else None
    final_eval_std = float(eval_df["eval_std_reward"].iloc[-1]) if not eval_df.empty else None
    best_eval_mean = float(eval_df["eval_mean_reward"].max()) if not eval_df.empty else None

    return {
        "optimizer": optimizer_name,
        "seed": seed,
        "optimizer_config": json.dumps(spec, sort_keys=True),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "total_steps": total_steps,
        "elapsed_seconds": time.time() - training_start,
        "last_20_train_reward_mean": last_20_mean,
        "best_train_reward": float(history_df["episode_reward"].max()),
        "final_eval_mean_reward": final_eval_mean,
        "final_eval_std_reward": final_eval_std,
        "best_eval_mean_reward": best_eval_mean,
        "model_path": str(model_path),
        "history_path": str(history_path),
        "eval_path": str(eval_path),
        "shift_eval_path": str(shift_path),
    }


def build_summary(raw_results: pd.DataFrame):
    summary = (
        raw_results.groupby("optimizer", as_index=False)
        .agg(
            num_runs=("seed", "count"),
            final_eval_mean_reward_mean=("final_eval_mean_reward", "mean"),
            final_eval_mean_reward_std=("final_eval_mean_reward", "std"),
            best_eval_mean_reward_mean=("best_eval_mean_reward", "mean"),
            best_eval_mean_reward_std=("best_eval_mean_reward", "std"),
            last_20_train_reward_mean_mean=("last_20_train_reward_mean", "mean"),
            last_20_train_reward_mean_std=("last_20_train_reward_mean", "std"),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
            elapsed_seconds_std=("elapsed_seconds", "std"),
        )
        .sort_values("final_eval_mean_reward_mean", ascending=False)
    )
    return summary


def write_manifest(run_dir: Path, args):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "env_name": ENV_NAME,
        "torch_version": torch.__version__,
        "device": args.device,
        "args": vars(args),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def build_shift_summary(shift_results: pd.DataFrame):
    return (
        shift_results.groupby(["shift_type", "shift_value", "optimizer"], as_index=False)
        .agg(
            num_runs=("seed", "count"),
            mean_reward_mean=("mean_reward", "mean"),
            mean_reward_std=("mean_reward", "std"),
            mean_within_run_std=("std_reward", "mean"),
        )
        .sort_values(["shift_type", "shift_value", "mean_reward_mean"], ascending=[True, True, False])
    )


def main():
    args = parse_args()
    run_dir = Path(args.output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    write_manifest(run_dir, args)

    raw_results = []
    for optimizer_name in args.optimizers:
        for seed in args.seeds:
            result = run_single_experiment(optimizer_name, seed, args, run_dir, device)
            raw_results.append(result)
            pd.DataFrame(raw_results).to_csv(run_dir / "raw_results.csv", index=False)

    raw_results_df = pd.DataFrame(raw_results)
    summary_df = build_summary(raw_results_df)
    summary_df.to_csv(run_dir / "summary_by_optimizer.csv", index=False)

    history_frames = []
    eval_frames = []
    shift_frames = []
    for history_file in sorted((run_dir / "histories").glob("*.csv")):
        history_frames.append(pd.read_csv(history_file))
    for eval_file in sorted((run_dir / "evals").glob("*.csv")):
        eval_frames.append(pd.read_csv(eval_file))
    for shift_file in sorted((run_dir / "shift_evals").glob("*.csv")):
        shift_frames.append(pd.read_csv(shift_file))

    all_histories = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    all_evals = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
    all_shift_evals = pd.concat(shift_frames, ignore_index=True) if shift_frames else pd.DataFrame()
    all_histories.to_csv(run_dir / "all_histories.csv", index=False)
    all_evals.to_csv(run_dir / "all_evals.csv", index=False)
    all_shift_evals.to_csv(run_dir / "all_shift_evals.csv", index=False)
    if not all_shift_evals.empty:
        build_shift_summary(all_shift_evals).to_csv(run_dir / "summary_shifted_envs.csv", index=False)

    if args.save_plots:
        maybe_save_plots(run_dir, all_histories, summary_df)

    print(f"Run directory: {run_dir}")
    print(f"Raw results: {run_dir / 'raw_results.csv'}")
    print(f"Summary: {run_dir / 'summary_by_optimizer.csv'}")
    print(f"Histories: {run_dir / 'all_histories.csv'}")
    print(f"Evals: {run_dir / 'all_evals.csv'}")
    print(f"Shifted evals: {run_dir / 'all_shift_evals.csv'}")
    if not all_shift_evals.empty:
        print(f"Shift summary: {run_dir / 'summary_shifted_envs.csv'}")


if __name__ == "__main__":
    main()
