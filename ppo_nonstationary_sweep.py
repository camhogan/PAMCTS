import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical

from ddqn_nonstationary_sweep import (
    HybridOptimizer,
    Muon,
    build_baseline_env_kwargs,
    build_optimizer,
    build_shift_env_kwargs,
    build_shift_summary,
    build_training_summary,
    generate_auto_plots,
    get_shift_settings,
    get_task,
    optimizer_spec,
    set_all_seeds,
    write_manifest,
)


@dataclass
class RolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    next_value: torch.Tensor


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x, action=None):
        logits, value = self(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def act_deterministic(self, x):
        logits, value = self(x)
        action = torch.argmax(logits, dim=-1)
        return action, value


def split_muon_param_groups_actor_critic(model: nn.Module):
    muon_named_params = []
    fallback_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Keep the shared trunk on Muon and route both output heads through AdamW.
        if param.ndim >= 2 and not name.startswith(("policy_head", "value_head")):
            muon_named_params.append((name, param))
        else:
            fallback_params.append((name, param))
    return muon_named_params, fallback_params


def build_ppo_optimizer(parameters, spec, model, args):
    if spec["name"] != "muon":
        return build_optimizer(
            parameters,
            spec,
            model=model,
            muon_spectrum_every=args.muon_spectrum_every,
            muon_spectrum_topk=args.muon_spectrum_topk,
            adamw_momentum_every=(
                args.adamw_momentum_every
                if spec["name"] != "muon"
                else max(args.adamw_momentum_every, args.muon_adamw_momentum_every)
            ),
            sgd_momentum_every=args.sgd_momentum_log_every,
        )

    muon_named_params, fallback_named_params = split_muon_param_groups_actor_critic(model)
    muon_params = [param for _name, param in muon_named_params]
    fallback_params = [param for _name, param in fallback_named_params]

    muon_opt = (
        Muon(
            muon_params,
            lr=spec["lr"],
            momentum=spec["momentum"],
            ns_steps=spec["ns_steps"],
            weight_decay=spec["weight_decay"],
            param_name_map={id(param): name for name, param in muon_named_params},
            spectrum_every=args.muon_spectrum_every,
            spectrum_topk=args.muon_spectrum_topk,
        )
        if muon_params
        else None
    )
    fallback_opt = (
        torch.optim.AdamW(
            fallback_params,
            lr=spec["fallback_lr"],
            weight_decay=spec["fallback_weight_decay"],
        )
        if fallback_params
        else None
    )
    return HybridOptimizer(muon_opt, fallback_opt)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO baselines and evaluate them under non-stationary settings.")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["cartpole", "frozenlake"],
        choices=["cartpole", "frozenlake"],
        help="Task families to include in the sweep.",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["sgd", "sgd_momentum", "sgd_nag", "adam", "adamw", "rmsprop", "muon"],
        choices=["sgd", "sgd_momentum", "sgd_nag", "adam", "adamw", "rmsprop", "muon"],
        help="Optimizers to compare.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(30)), help="Seeds to run.")
    parser.add_argument(
        "--cartpole-episodes",
        type=int,
        default=0,
        help="Maximum training episodes per CartPole run (0 disables episode cap).",
    )
    parser.add_argument(
        "--frozenlake-episodes",
        type=int,
        default=0,
        help="Maximum training episodes per FrozenLake run (0 disables episode cap).",
    )
    parser.add_argument("--cartpole-train-steps", type=int, default=100000, help="Environment-step training budget for CartPole.")
    parser.add_argument("--frozenlake-train-steps", type=int, default=1000000, help="Environment-step training budget for FrozenLake.")
    parser.add_argument("--cartpole-max-steps", type=int, default=2500, help="CartPole step cap per episode.")
    parser.add_argument("--frozenlake-max-steps", type=int, default=200, help="FrozenLake step cap per episode.")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Baseline evaluation episodes per checkpoint/final run.")
    parser.add_argument("--shift-eval-episodes", type=int, default=20, help="Episodes per shifted-environment evaluation.")
    parser.add_argument("--eval-every", type=int, default=25, help="Evaluate on the baseline every N completed training episodes.")
    parser.add_argument("--rollout-steps", type=int, default=512, help="PPO rollout horizon in environment steps.")
    parser.add_argument("--update-epochs", type=int, default=4, help="Number of PPO epochs per rollout.")
    parser.add_argument("--minibatch-size", type=int, default=128, help="Minibatch size for PPO updates.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO ratio clipping coefficient.")
    parser.add_argument("--value-clip-coef", type=float, default=0.2, help="Optional value clipping coefficient.")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient.")
    parser.add_argument("--value-loss-coef", type=float, default=0.5, help="Value-loss coefficient.")
    parser.add_argument("--normalize-advantages", action=argparse.BooleanOptionalAction, default=True, help="Normalize advantages per update.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping max norm.")
    parser.add_argument("--shared-lr", type=float, default=None, help="Override learning rate for all optimizers.")
    parser.add_argument("--sgd-lr", type=float, default=0.01, help="Learning rate for plain SGD.")
    parser.add_argument("--sgd-momentum-lr", type=float, default=0.01, help="Learning rate for SGD with momentum.")
    parser.add_argument("--sgd-nag-lr", type=float, default=0.01, help="Learning rate for SGD with Nesterov momentum.")
    parser.add_argument("--adam-lr", type=float, default=0.0003, help="Learning rate for Adam.")
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="Beta1 (first-moment momentum) for Adam.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="Beta2 (second-moment decay) for Adam.")
    parser.add_argument("--adamw-lr", type=float, default=0.0003, help="Learning rate for AdamW.")
    parser.add_argument("--adamw-beta1", type=float, default=0.9, help="Beta1 (first-moment momentum) for AdamW.")
    parser.add_argument("--adamw-beta2", type=float, default=0.999, help="Beta2 (second-moment decay) for AdamW.")
    parser.add_argument("--rmsprop-lr", type=float, default=0.0007, help="Learning rate for RMSprop.")
    parser.add_argument("--muon-lr", type=float, default=0.0003, help="Learning rate for Muon.")
    parser.add_argument(
        "--muon-adamw-lr",
        type=float,
        default=None,
        help="Learning rate for AdamW fallback parameters when using Muon (defaults to Muon LR).",
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum used for SGD variants.")
    parser.add_argument("--muon-momentum", type=float, default=0.95, help="Momentum used for Muon.")
    parser.add_argument("--muon-ns-steps", type=int, default=5, help="Newton-Schulz iterations for Muon.")
    parser.add_argument("--muon-weight-decay", type=float, default=0.0, help="Decoupled weight decay for Muon.")
    parser.add_argument(
        "--muon-adamw-weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for AdamW fallback parameters when using Muon.",
    )
    parser.add_argument(
        "--muon-spectrum-every",
        type=int,
        default=0,
        help="Log Muon raw/orthogonalized update singular values every N optimizer steps (0 disables).",
    )
    parser.add_argument(
        "--muon-spectrum-topk",
        type=int,
        default=0,
        help="Keep only top-k singular values when logging Muon update spectra (0 keeps all).",
    )
    parser.add_argument(
        "--adamw-momentum-every",
        type=int,
        default=0,
        help="Log AdamW exp_avg momentum stats every N optimizer steps for AdamW/Muon-hybrid runs (0 disables).",
    )
    parser.add_argument(
        "--muon-adamw-momentum-every",
        type=int,
        default=0,
        help="Log AdamW exp_avg momentum stats every N optimizer steps when using Muon hybrid (0 disables).",
    )
    parser.add_argument(
        "--sgd-momentum-log-every",
        type=int,
        default=0,
        help="Log SGD momentum-buffer stats every N optimizer steps for sgd_momentum/sgd_nag runs (0 disables).",
    )
    parser.add_argument(
        "--auto-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically generate standard plots at run end.",
    )
    parser.add_argument(
        "--auto-plot-step-bin",
        type=int,
        default=100,
        help="Step-bin size for auto-generated comparison training plots (0 disables binning).",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for Adam/AdamW.")
    parser.add_argument("--cartpole-gamma", type=float, default=0.99, help="Discount factor for CartPole.")
    parser.add_argument("--frozenlake-gamma", type=float, default=0.95, help="Discount factor for FrozenLake.")
    parser.add_argument("--cartpole-gravity-settings", nargs="+", type=float, default=[9.8, 20.0, 50.0, 500.0], help="CartPole gravity shifts.")
    parser.add_argument("--cartpole-masscart-settings", nargs="+", type=float, default=[0.1, 1.0, 1.2, 1.3, 1.5], help="CartPole masscart shifts.")
    parser.add_argument("--output-dir", default="results/ppo_nonstationary_sweep", help="Sweep output directory.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    return parser.parse_args()


def get_domain_hyperparams(name: str, args):
    if name == "cartpole":
        return {
            "episode_cap": args.cartpole_episodes,
            "train_steps": args.cartpole_train_steps,
            "gamma": args.cartpole_gamma,
        }
    if name == "frozenlake":
        return {
            "episode_cap": args.frozenlake_episodes,
            "train_steps": args.frozenlake_train_steps,
            "gamma": args.frozenlake_gamma,
        }
    raise ValueError(f"Unsupported domain: {name}")


def compute_gae(batch: RolloutBatch, gamma: float, gae_lambda: float):
    rewards = batch.rewards
    dones = batch.dones
    values = batch.values
    next_value = batch.next_value
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(1, device=rewards.device, dtype=rewards.dtype)

    for step in reversed(range(rewards.shape[0])):
        if step == rewards.shape[0] - 1:
            next_nonterminal = 1.0 - dones[step]
            next_values = next_value
        else:
            next_nonterminal = 1.0 - dones[step]
            next_values = values[step + 1]
        delta = rewards[step] + gamma * next_values * next_nonterminal - values[step]
        last_advantage = delta + gamma * gae_lambda * next_nonterminal * last_advantage
        advantages[step] = last_advantage

    returns = advantages + values
    return advantages, returns


def evaluate_on_task(policy, task, env_kwargs, episodes, seed, device):
    rewards = []
    for episode in range(episodes):
        env_seed = seed + 10_000 + episode
        env = task.make_env(env_seed, **env_kwargs)
        state = task.reset(env, env_seed)
        episode_reward = 0.0
        for _ in range(task.max_steps):
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action, _value = policy.act_deterministic(state_tensor)
            state, reward, done = task.step(env, int(action.item()))
            episode_reward += reward
            if done:
                break
        rewards.append(episode_reward)
        env.close()
    return rewards


def collect_rollout(policy, env, task, current_state, current_seed, total_steps, train_steps, device, rollout_steps, episode_cap):
    states = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []
    episode_completions = []
    running_episode_reward = collect_rollout.running_episode_reward
    completed_episodes = collect_rollout.completed_episodes

    steps_to_collect = min(rollout_steps, train_steps - total_steps)
    for _ in range(steps_to_collect):
        if episode_cap > 0 and completed_episodes >= episode_cap:
            break

        state_tensor = torch.tensor(current_state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _entropy, value = policy.get_action_and_value(state_tensor)

        next_state, reward, done = task.step(env, int(action.item()))
        states.append(state_tensor.squeeze(0))
        actions.append(action.squeeze(0))
        log_probs.append(log_prob.squeeze(0))
        rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
        dones.append(torch.tensor(float(done), dtype=torch.float32, device=device))
        values.append(value.squeeze(0))

        total_steps += 1
        running_episode_reward += reward
        current_state = next_state

        if done:
            completed_episodes += 1
            episode_completions.append(
                {
                    "episode": completed_episodes,
                    "episode_reward": running_episode_reward,
                    "total_steps": total_steps,
                }
            )
            running_episode_reward = 0.0
            if episode_cap > 0 and completed_episodes >= episode_cap:
                break
            current_seed += 1
            current_state = task.reset(env, current_seed)

        if total_steps >= train_steps:
            break

    with torch.no_grad():
        next_state_tensor = torch.tensor(current_state, dtype=torch.float32, device=device).unsqueeze(0)
        _next_action, _next_log_prob, _next_entropy, next_value = policy.get_action_and_value(next_state_tensor)

    collect_rollout.running_episode_reward = running_episode_reward
    collect_rollout.completed_episodes = completed_episodes

    batch = RolloutBatch(
        states=torch.stack(states),
        actions=torch.stack(actions).long(),
        log_probs=torch.stack(log_probs),
        rewards=torch.stack(rewards),
        dones=torch.stack(dones),
        values=torch.stack(values),
        next_value=next_value.squeeze(0),
    )
    return batch, current_state, current_seed, total_steps, episode_completions


collect_rollout.running_episode_reward = 0.0
collect_rollout.completed_episodes = 0


def ppo_update(policy, optimizer, batch: RolloutBatch, advantages, returns, args):
    batch_size = batch.states.shape[0]
    minibatch_size = min(args.minibatch_size, batch_size)
    indices = np.arange(batch_size)
    losses = []

    if args.normalize_advantages and batch_size > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    b_states = batch.states
    b_actions = batch.actions
    b_log_probs = batch.log_probs
    b_values = batch.values

    for _ in range(args.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, batch_size, minibatch_size):
            batch_idx = torch.as_tensor(indices[start : start + minibatch_size], device=b_states.device, dtype=torch.long)

            _, new_log_probs, entropy, new_values = policy.get_action_and_value(
                b_states[batch_idx],
                b_actions[batch_idx],
            )
            log_ratio = new_log_probs - b_log_probs[batch_idx]
            ratio = log_ratio.exp()

            mb_advantages = advantages[batch_idx]
            pg_loss_1 = -mb_advantages * ratio
            pg_loss_2 = -mb_advantages * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
            policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()

            value_pred = new_values
            if args.value_clip_coef > 0:
                value_delta = value_pred - b_values[batch_idx]
                value_pred_clipped = b_values[batch_idx] + torch.clamp(
                    value_delta,
                    -args.value_clip_coef,
                    args.value_clip_coef,
                )
                value_loss_unclipped = (value_pred - returns[batch_idx]) ** 2
                value_loss_clipped = (value_pred_clipped - returns[batch_idx]) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            else:
                value_loss = 0.5 * ((value_pred - returns[batch_idx]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = policy_loss + args.value_loss_coef * value_loss - args.entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()

            losses.append(
                {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy_loss.item()),
                }
            )

    return losses


def run_single_experiment(domain_name: str, optimizer_name: str, seed: int, args, run_dir: Path, device: torch.device):
    task = get_task(domain_name, args)
    hyperparams = get_domain_hyperparams(domain_name, args)
    episode_cap = hyperparams["episode_cap"]
    train_steps = hyperparams["train_steps"]
    gamma = hyperparams["gamma"]
    set_all_seeds(seed)
    collect_rollout.running_episode_reward = 0.0
    collect_rollout.completed_episodes = 0

    env = task.make_env(seed, **build_baseline_env_kwargs(domain_name))
    current_seed = seed
    current_state = task.reset(env, current_seed)
    policy = ActorCriticNetwork(task.state_dim, task.action_dim).to(device)
    spec = optimizer_spec(optimizer_name, args)
    optimizer = build_ppo_optimizer(
        policy.parameters(),
        spec,
        model=policy,
        args=args,
    )

    history_rows = []
    eval_rows = []
    shift_rows = []
    total_steps = 0
    start_time = time.time()
    last_eval_episode = 0

    print(
        f"[start] domain={domain_name} optimizer={optimizer_name} seed={seed} "
        f"train_steps={train_steps} episode_cap={episode_cap}",
        flush=True,
    )

    while total_steps < train_steps:
        if episode_cap > 0 and collect_rollout.completed_episodes >= episode_cap:
            break

        batch, current_state, current_seed, total_steps, episode_completions = collect_rollout(
            policy,
            env,
            task,
            current_state,
            current_seed,
            total_steps,
            train_steps,
            device,
            args.rollout_steps,
            episode_cap,
        )
        if batch.states.shape[0] == 0:
            break

        advantages, returns = compute_gae(batch, gamma, args.gae_lambda)
        losses = ppo_update(policy, optimizer, batch, advantages, returns, args)

        loss_summary = {
            "mean_loss": float(np.mean([row["loss"] for row in losses])) if losses else None,
            "mean_policy_loss": float(np.mean([row["policy_loss"] for row in losses])) if losses else None,
            "mean_value_loss": float(np.mean([row["value_loss"] for row in losses])) if losses else None,
            "mean_entropy": float(np.mean([row["entropy"] for row in losses])) if losses else None,
        }

        for completion in episode_completions:
            history_rows.append(
                {
                    "domain": domain_name,
                    "optimizer": optimizer_name,
                    "seed": seed,
                    "episode": completion["episode"],
                    "episode_reward": completion["episode_reward"],
                    "total_steps": completion["total_steps"],
                    "mean_loss": loss_summary["mean_loss"],
                    "mean_policy_loss": loss_summary["mean_policy_loss"],
                    "mean_value_loss": loss_summary["mean_value_loss"],
                    "mean_entropy": loss_summary["mean_entropy"],
                }
            )
            if completion["episode"] % args.eval_every == 0:
                baseline_rewards = evaluate_on_task(
                    policy,
                    task,
                    build_baseline_env_kwargs(domain_name),
                    args.eval_episodes,
                    seed + completion["episode"],
                    device,
                )
                eval_rows.append(
                    {
                        "domain": domain_name,
                        "optimizer": optimizer_name,
                        "seed": seed,
                        "episode": completion["episode"],
                        "total_steps": completion["total_steps"],
                        "eval_mean_reward": float(np.mean(baseline_rewards)),
                        "eval_std_reward": float(np.std(baseline_rewards)),
                    }
                )
                last_eval_episode = completion["episode"]
                print(
                    f"[progress] domain={domain_name} optimizer={optimizer_name} seed={seed} "
                    f"episode={completion['episode']} steps={completion['total_steps']}/{train_steps} "
                    f"train_reward={completion['episode_reward']:.3f} baseline_eval_mean={np.mean(baseline_rewards):.3f}",
                    flush=True,
                )

        if episode_cap > 0 and collect_rollout.completed_episodes >= episode_cap:
            break

    if last_eval_episode != collect_rollout.completed_episodes or not eval_rows:
        baseline_rewards = evaluate_on_task(
            policy,
            task,
            build_baseline_env_kwargs(domain_name),
            args.eval_episodes,
            seed + collect_rollout.completed_episodes,
            device,
        )
        eval_rows.append(
            {
                "domain": domain_name,
                "optimizer": optimizer_name,
                "seed": seed,
                "episode": collect_rollout.completed_episodes,
                "total_steps": total_steps,
                "eval_mean_reward": float(np.mean(baseline_rewards)),
                "eval_std_reward": float(np.std(baseline_rewards)),
            }
        )

    print(
        f"[shift-eval] domain={domain_name} optimizer={optimizer_name} seed={seed} starting shifted evaluations",
        flush=True,
    )
    for shift_setting in get_shift_settings(domain_name, task, args):
        rewards = evaluate_on_task(
            policy,
            task,
            build_shift_env_kwargs(domain_name, shift_setting),
            args.shift_eval_episodes,
            seed + 50_000,
            device,
        )
        shift_rows.append(
            {
                "domain": domain_name,
                "optimizer": optimizer_name,
                "seed": seed,
                "shift_family": shift_setting["shift_family"],
                "shift_value": shift_setting["shift_value"],
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
            }
        )
        print(
            f"[shift] domain={domain_name} optimizer={optimizer_name} seed={seed} "
            f"{shift_setting['shift_family']}={shift_setting['shift_value']} mean_reward={np.mean(rewards):.3f}",
            flush=True,
        )

    env.close()

    history_df = pd.DataFrame(history_rows)
    eval_df = pd.DataFrame(eval_rows)
    shift_df = pd.DataFrame(shift_rows)

    model_path = run_dir / "models" / f"{domain_name}_{optimizer_name}_seed{seed}.pt"
    history_path = run_dir / "histories" / f"{domain_name}_{optimizer_name}_seed{seed}.csv"
    eval_path = run_dir / "evals" / f"{domain_name}_{optimizer_name}_seed{seed}.csv"
    shift_path = run_dir / "shift_evals" / f"{domain_name}_{optimizer_name}_seed{seed}.csv"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    shift_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), model_path)
    history_df.to_csv(history_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    shift_df.to_csv(shift_path, index=False)

    result = {
        "domain": domain_name,
        "optimizer": optimizer_name,
        "seed": seed,
        "optimizer_config": json.dumps(spec, sort_keys=True),
        "episodes": int(collect_rollout.completed_episodes),
        "episode_cap": episode_cap,
        "train_steps_budget": train_steps,
        "max_steps": task.max_steps,
        "total_steps": total_steps,
        "elapsed_seconds": time.time() - start_time,
        "last_20_train_reward_mean": float(history_df["episode_reward"].tail(20).mean()) if not history_df.empty else None,
        "best_train_reward": float(history_df["episode_reward"].max()) if not history_df.empty else None,
        "final_eval_mean_reward": float(eval_df["eval_mean_reward"].iloc[-1]) if not eval_df.empty else None,
        "best_eval_mean_reward": float(eval_df["eval_mean_reward"].max()) if not eval_df.empty else None,
        "model_path": str(model_path),
        "history_path": str(history_path),
        "eval_path": str(eval_path),
        "shift_eval_path": str(shift_path),
    }
    print(
        f"[done] domain={domain_name} optimizer={optimizer_name} seed={seed} "
        f"elapsed={result['elapsed_seconds']:.1f}s final_eval={result['final_eval_mean_reward']}",
        flush=True,
    )
    return result


def main():
    args = parse_args()
    run_dir = Path(args.output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    write_manifest(run_dir, args)

    raw_results = []
    total_runs = len(args.domains) * len(args.optimizers) * len(args.seeds)
    run_index = 0
    for domain_name in args.domains:
        for optimizer_name in args.optimizers:
            for seed in args.seeds:
                run_index += 1
                print(
                    f"[queue] run={run_index}/{total_runs} domain={domain_name} optimizer={optimizer_name} seed={seed}",
                    flush=True,
                )
                result = run_single_experiment(domain_name, optimizer_name, seed, args, run_dir, device)
                raw_results.append(result)
                pd.DataFrame(raw_results).to_csv(run_dir / "raw_results.csv", index=False)

    raw_results_df = pd.DataFrame(raw_results)
    build_training_summary(raw_results_df).to_csv(run_dir / "summary_training.csv", index=False)

    history_frames = [pd.read_csv(path) for path in sorted((run_dir / "histories").glob("*.csv"))]
    eval_frames = [pd.read_csv(path) for path in sorted((run_dir / "evals").glob("*.csv"))]
    shift_frames = [pd.read_csv(path) for path in sorted((run_dir / "shift_evals").glob("*.csv"))]

    all_histories = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    all_evals = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
    all_shift_evals = pd.concat(shift_frames, ignore_index=True) if shift_frames else pd.DataFrame()

    all_histories.to_csv(run_dir / "all_histories.csv", index=False)
    all_evals.to_csv(run_dir / "all_evals.csv", index=False)
    all_shift_evals.to_csv(run_dir / "all_shift_evals.csv", index=False)
    if not all_shift_evals.empty:
        build_shift_summary(all_shift_evals).to_csv(run_dir / "summary_shifted_envs.csv", index=False)

    print(f"Run directory: {run_dir}")
    print(f"Raw results: {run_dir / 'raw_results.csv'}")
    print(f"Training summary: {run_dir / 'summary_training.csv'}")
    print(f"Histories: {run_dir / 'all_histories.csv'}")
    print(f"Baseline evals: {run_dir / 'all_evals.csv'}")
    print(f"Shifted evals: {run_dir / 'all_shift_evals.csv'}")
    if not all_shift_evals.empty:
        print(f"Shift summary: {run_dir / 'summary_shifted_envs.csv'}")
    if args.auto_plot:
        generate_auto_plots(run_dir, args)


if __name__ == "__main__":
    main()
