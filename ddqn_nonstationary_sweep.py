import argparse
import csv
import json
import random
import subprocess
import sys
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


CARTPOLE_ENV_NAME = "CartPole-v1"
FROZENLAKE_MAP = ["SHF", "FFF", "HFG"]
FROZENLAKE_BASELINE = (1.0, 0.0, 0.0)
FROZENLAKE_SHIFT_SETTINGS = [
    (1.0, 0.0, 0.0),
    (0.833, 0.083, 0.083),
    (0.633, 0.183, 0.183),
    (0.433, 0.283, 0.283),
    (0.333, 0.333, 0.333),
]


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


class Muon(torch.optim.Optimizer):
    """Single-device Muon optimizer aligned with Keller Jordan's reference logic."""

    def __init__(
        self,
        params,
        lr=0.001,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        eps=1e-8,
        weight_decay=0.0,
        param_name_map=None,
        spectrum_logger=None,
        spectrum_every=0,
        spectrum_topk=0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self.param_name_map = param_name_map or {}
        self.spectrum_logger = spectrum_logger
        self.spectrum_every = max(0, int(spectrum_every))
        self.spectrum_topk = int(spectrum_topk)
        self._step = 0

    @staticmethod
    def _zeropower_via_newtonschulz5(grad, steps: int):
        """Reference NS5 orthogonalization used by Muon."""
        assert grad.ndim >= 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        x = grad.bfloat16()
        if grad.size(-2) > grad.size(-1):
            x = x.mT
        x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(steps):
            a_mat = x @ x.mT
            b_mat = b * a_mat + c * a_mat @ a_mat
            x = a * x + b_mat @ x
        if grad.size(-2) > grad.size(-1):
            x = x.mT
        return x

    @classmethod
    def _muon_update(cls, grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
        momentum.lerp_(grad, 1 - beta)
        update = grad.lerp_(momentum, beta) if nesterov else momentum
        if update.ndim < 2:
            return update, None, None
        if update.ndim == 4:
            update = update.view(len(update), -1)
        raw_update = update.detach().float()
        ortho_update = cls._zeropower_via_newtonschulz5(update, steps=ns_steps)
        ortho_update *= max(1, ortho_update.size(-2) / ortho_update.size(-1)) ** 0.5
        return ortho_update.to(dtype=grad.dtype), raw_update, ortho_update.detach().float()

    def _log_update_spectrum(self, param, raw_update, ortho_update):
        if self.spectrum_logger is None:
            return

        param_name = self.param_name_map.get(id(param), "<unnamed>")
        for update_kind, update in (("raw_update", raw_update), ("ortho_update", ortho_update)):
            full_values = torch.linalg.svdvals(update).cpu().tolist()
            sigma_max = float(full_values[0]) if full_values else None
            sigma_min = float(full_values[-1]) if full_values else None
            if sigma_max is None or sigma_min is None:
                condition_number = None
            elif sigma_min <= 1e-12:
                condition_number = float("inf")
            else:
                condition_number = float(sigma_max / sigma_min)
            values = full_values
            if self.spectrum_topk > 0:
                values = values[: self.spectrum_topk]
            self.spectrum_logger(
                {
                    "optimizer_step": self._step,
                    "param_name": param_name,
                    "update_kind": update_kind,
                    "singular_values": values,
                    "sigma_max": sigma_max,
                    "sigma_min": sigma_min,
                    "condition_number": condition_number,
                }
            )

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step += 1
        log_spectrum = self.spectrum_logger is not None and self.spectrum_every > 0 and self._step % self.spectrum_every == 0

        for group in self.param_groups:
            lr, beta = group["lr"], group["momentum"]
            nesterov, ns_steps = group["nesterov"], group["ns_steps"]
            wd = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)

                if wd > 0:
                    param.data.mul_(1.0 - lr * wd)

                state = self.state[param]
                buf = state.setdefault("momentum_buffer", torch.zeros_like(param))
                update, raw_update, ortho_update = self._muon_update(
                    param.grad,
                    buf,
                    beta=beta,
                    ns_steps=ns_steps,
                    nesterov=nesterov,
                )
                if log_spectrum and raw_update is not None and ortho_update is not None:
                    self._log_update_spectrum(param, raw_update, ortho_update)
                param.data.add_(update.reshape(param.shape), alpha=-lr)

        return loss


class HybridOptimizer:
    """Thin wrapper that steps multiple optimizers as one."""

    def __init__(
        self,
        *optimizers,
        adamw_momentum_logger=None,
        adamw_momentum_every=0,
        adamw_param_name_map=None,
        sgd_momentum_logger=None,
        sgd_momentum_every=0,
        sgd_param_name_map=None,
    ):
        self.optimizers = [opt for opt in optimizers if opt is not None]
        if not self.optimizers:
            raise ValueError("HybridOptimizer requires at least one inner optimizer")
        self.adamw_momentum_logger = adamw_momentum_logger
        self.adamw_momentum_every = max(0, int(adamw_momentum_every))
        self.adamw_param_name_map = adamw_param_name_map or {}
        self.sgd_momentum_logger = sgd_momentum_logger
        self.sgd_momentum_every = max(0, int(sgd_momentum_every))
        self.sgd_param_name_map = sgd_param_name_map or {}
        self._step = 0

    @property
    def param_groups(self):
        groups = []
        for optimizer in self.optimizers:
            groups.extend(optimizer.param_groups)
        return groups

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        self._step += 1
        for optimizer in self.optimizers:
            optimizer.step()
        if (
            self.adamw_momentum_logger is not None
            and self.adamw_momentum_every > 0
            and self._step % self.adamw_momentum_every == 0
        ):
            self._log_adamw_momentum()
        if (
            self.sgd_momentum_logger is not None
            and self.sgd_momentum_every > 0
            and self._step % self.sgd_momentum_every == 0
        ):
            self._log_sgd_momentum()
        return loss

    def state_dict(self):
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict):
        optimizer_states = state_dict.get("optimizers", [])
        if len(optimizer_states) != len(self.optimizers):
            raise ValueError(
                f"Expected {len(self.optimizers)} optimizer states, got {len(optimizer_states)}"
            )
        for optimizer, optimizer_state in zip(self.optimizers, optimizer_states):
            optimizer.load_state_dict(optimizer_state)

    def _log_adamw_momentum(self):
        for optimizer in self.optimizers:
            if not isinstance(optimizer, torch.optim.AdamW):
                continue
            for group in optimizer.param_groups:
                beta1, beta2 = group["betas"]
                eps = float(group["eps"])
                weight_decay = float(group.get("weight_decay", 0.0))
                for param in group["params"]:
                    state = optimizer.state.get(param)
                    if not state:
                        continue
                    exp_avg = state.get("exp_avg")
                    if exp_avg is None:
                        continue
                    exp_avg_f = exp_avg.detach().float()
                    exp_avg_sq = state.get("exp_avg_sq")
                    exp_avg_sq_f = exp_avg_sq.detach().float() if exp_avg_sq is not None else None
                    exp_avg_sq_l2 = float(exp_avg_sq_f.norm().item()) if exp_avg_sq_f is not None else None

                    state_step = state.get("step", 0)
                    if isinstance(state_step, torch.Tensor):
                        step = int(state_step.item())
                    else:
                        step = int(state_step)
                    step = max(step, 1)
                    bias_correction1 = 1.0 - beta1**step
                    bias_correction2 = 1.0 - beta2**step

                    m_hat = exp_avg_f / bias_correction1
                    if exp_avg_sq_f is not None:
                        v_hat = exp_avg_sq_f / bias_correction2
                        sqrt_v_hat = v_hat.sqrt()
                        adaptive_term = m_hat / (sqrt_v_hat + eps)
                        wd_term = weight_decay * param.detach().float()
                        parenthesized_term = adaptive_term + wd_term
                        sqrt_v_hat_l2 = float(sqrt_v_hat.norm().item())
                        parenthesized_term_l2 = float(parenthesized_term.norm().item())
                    else:
                        sqrt_v_hat_l2 = None
                        parenthesized_term_l2 = None

                    self.adamw_momentum_logger(
                        {
                            "optimizer_step": self._step,
                            "param_name": self.adamw_param_name_map.get(id(param), "<unnamed>"),
                            "exp_avg_l2": float(exp_avg_f.norm().item()),
                            "exp_avg_mean_abs": float(exp_avg_f.abs().mean().item()),
                            "exp_avg_max_abs": float(exp_avg_f.abs().max().item()),
                            "exp_avg_rms": float(exp_avg_f.pow(2).mean().sqrt().item()),
                            "exp_avg_sq_l2": exp_avg_sq_l2,
                            "m_hat_l2": float(m_hat.norm().item()),
                            "sqrt_v_hat_l2": sqrt_v_hat_l2,
                            "parenthesized_term_l2": parenthesized_term_l2,
                        }
                    )

    def _log_sgd_momentum(self):
        for optimizer in self.optimizers:
            if not isinstance(optimizer, torch.optim.SGD):
                continue
            for group in optimizer.param_groups:
                for param in group["params"]:
                    state = optimizer.state.get(param)
                    if not state:
                        continue
                    momentum_buffer = state.get("momentum_buffer")
                    if momentum_buffer is None:
                        continue
                    mb = momentum_buffer.detach().float()
                    self.sgd_momentum_logger(
                        {
                            "optimizer_step": self._step,
                            "param_name": self.sgd_param_name_map.get(id(param), "<unnamed>"),
                            "momentum_buffer_l2": float(mb.norm().item()),
                            "momentum_buffer_mean_abs": float(mb.abs().mean().item()),
                            "momentum_buffer_max_abs": float(mb.abs().max().item()),
                            "momentum_buffer_rms": float(mb.pow(2).mean().sqrt().item()),
                        }
                    )


class MuonSpectrumCSVLogger:
    def __init__(self, path: Path, domain: str, optimizer: str, seed: int):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.domain = domain
        self.optimizer = optimizer
        self.seed = seed
        self.handle = self.path.open("w", newline="")
        self.writer = csv.writer(self.handle)
        self.writer.writerow(
            [
                "domain",
                "optimizer",
                "seed",
                "optimizer_step",
                "param_name",
                "update_kind",
                "sigma_max",
                "sigma_min",
                "condition_number",
                "singular_index",
                "singular_value",
            ]
        )

    def __call__(self, row):
        for idx, value in enumerate(row["singular_values"]):
            self.writer.writerow(
                [
                    self.domain,
                    self.optimizer,
                    self.seed,
                    row["optimizer_step"],
                    row["param_name"],
                    row["update_kind"],
                    row["sigma_max"],
                    row["sigma_min"],
                    row["condition_number"],
                    idx,
                    float(value),
                ]
            )

    def close(self):
        self.handle.close()


class AdamWMomentumCSVLogger:
    def __init__(self, path: Path, domain: str, optimizer: str, seed: int):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.domain = domain
        self.optimizer = optimizer
        self.seed = seed
        self.handle = self.path.open("w", newline="")
        self.writer = csv.writer(self.handle)
        self.writer.writerow(
            [
                "domain",
                "optimizer",
                "seed",
                "optimizer_step",
                "param_name",
                "exp_avg_l2",
                "exp_avg_mean_abs",
                "exp_avg_max_abs",
                "exp_avg_rms",
                "exp_avg_sq_l2",
                "m_hat_l2",
                "sqrt_v_hat_l2",
                "parenthesized_term_l2",
            ]
        )

    def __call__(self, row):
        self.writer.writerow(
            [
                self.domain,
                self.optimizer,
                self.seed,
                row["optimizer_step"],
                row["param_name"],
                row["exp_avg_l2"],
                row["exp_avg_mean_abs"],
                row["exp_avg_max_abs"],
                row["exp_avg_rms"],
                row["exp_avg_sq_l2"],
                row["m_hat_l2"],
                row["sqrt_v_hat_l2"],
                row["parenthesized_term_l2"],
            ]
        )

    def close(self):
        self.handle.close()


class SGDMomentumCSVLogger:
    def __init__(self, path: Path, domain: str, optimizer: str, seed: int):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.domain = domain
        self.optimizer = optimizer
        self.seed = seed
        self.handle = self.path.open("w", newline="")
        self.writer = csv.writer(self.handle)
        self.writer.writerow(
            [
                "domain",
                "optimizer",
                "seed",
                "optimizer_step",
                "param_name",
                "momentum_buffer_l2",
                "momentum_buffer_mean_abs",
                "momentum_buffer_max_abs",
                "momentum_buffer_rms",
            ]
        )

    def __call__(self, row):
        self.writer.writerow(
            [
                self.domain,
                self.optimizer,
                self.seed,
                row["optimizer_step"],
                row["param_name"],
                row["momentum_buffer_l2"],
                row["momentum_buffer_mean_abs"],
                row["momentum_buffer_max_abs"],
                row["momentum_buffer_rms"],
            ]
        )

    def close(self):
        self.handle.close()


class CartPoleTask:
    name = "cartpole"
    state_dim = 4
    action_dim = 2

    def __init__(self, max_steps: int):
        self.max_steps = max_steps

    def make_env(self, seed: int, gravity=None, masscart=None):
        env = gym.make(CARTPOLE_ENV_NAME)
        env._max_episode_steps = self.max_steps
        env.action_space.seed(seed)
        if gravity is not None:
            env.unwrapped.gravity = gravity
        if masscart is not None:
            env.unwrapped.masscart = masscart
            env.unwrapped.total_mass = env.unwrapped.masspole + env.unwrapped.masscart
            env.unwrapped.polemass_length = env.unwrapped.masspole * env.unwrapped.length
        return env

    def reset(self, env, seed):
        state, _info = env.reset(seed=seed)
        return np.asarray(state, dtype=np.float32)

    def step(self, env, action: int):
        next_state, reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated
        return np.asarray(next_state, dtype=np.float32), float(reward), done

    def baseline_eval_settings(self):
        return [{"shift_family": "baseline", "shift_value": "default", "gravity": None, "masscart": None}]

    def shift_settings(self, gravity_settings, masscart_settings):
        settings = []
        for gravity in gravity_settings:
            settings.append(
                {
                    "shift_family": "gravity",
                    "shift_value": f"{gravity:.3f}",
                    "gravity": gravity,
                    "masscart": None,
                }
            )
        for masscart in masscart_settings:
            settings.append(
                {
                    "shift_family": "masscart",
                    "shift_value": f"{masscart:.3f}",
                    "gravity": None,
                    "masscart": masscart,
                }
            )
        return settings


class FrozenLake3x3Task:
    name = "frozenlake"
    state_dim = 9
    action_dim = 4
    custom_map = FROZENLAKE_MAP
    left = 0
    down = 1
    right = 2
    up = 3

    def __init__(self, max_steps: int):
        self.max_steps = max_steps
        self.grid = np.asarray([list(row) for row in self.custom_map], dtype="U1")
        self.nrow, self.ncol = self.grid.shape
        self.start_state = 0
        self.goal_state = self._to_state(2, 2)
        self.holes = {self._to_state(0, 1), self._to_state(2, 0)}

    def _to_state(self, row: int, col: int):
        return row * self.ncol + col

    def _from_state(self, state: int):
        return divmod(state, self.ncol)

    def _move(self, state: int, action: int):
        row, col = self._from_state(state)
        if action == self.left:
            col = max(col - 1, 0)
        elif action == self.down:
            row = min(row + 1, self.nrow - 1)
        elif action == self.right:
            col = min(col + 1, self.ncol - 1)
        elif action == self.up:
            row = max(row - 1, 0)
        return self._to_state(row, col)

    def _one_hot(self, state: int):
        obs = np.zeros(self.state_dim, dtype=np.float32)
        obs[state] = 1.0
        return obs

    def make_env(self, seed: int, prob_distribution):
        return FrozenLake3x3Env(prob_distribution=prob_distribution, seed=seed, max_steps=self.max_steps, task=self)

    def reset(self, env, seed):
        return env.reset(seed=seed)

    def step(self, env, action: int):
        return env.step(action)

    def baseline_eval_settings(self):
        return [{"shift_family": "baseline", "shift_value": self._format_probs(FROZENLAKE_BASELINE), "prob_distribution": FROZENLAKE_BASELINE}]

    def shift_settings(self):
        return [
            {
                "shift_family": "prob_distribution",
                "shift_value": self._format_probs(setting),
                "prob_distribution": setting,
            }
            for setting in FROZENLAKE_SHIFT_SETTINGS
        ]

    def _format_probs(self, probs):
        return "[" + ", ".join(f"{prob:.3f}" for prob in probs) + "]"


class FrozenLake3x3Env:
    def __init__(self, prob_distribution, seed: int, max_steps: int, task: FrozenLake3x3Task):
        total = float(sum(prob_distribution))
        if total <= 0:
            raise ValueError(f"FrozenLake weights must sum to a positive value, got {prob_distribution}")
        self.prob_distribution = tuple(prob / total for prob in prob_distribution)
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.task = task
        self.step_count = 0
        self.state = self.task.start_state

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        self.step_count = 0
        self.state = self.task.start_state
        return self.task._one_hot(self.state)

    def step(self, action: int):
        straight_prob, clockwise_prob, counter_prob = self.prob_distribution
        roll = self.rng.random()
        if roll < straight_prob:
            actual_action = action
        elif roll < straight_prob + clockwise_prob:
            actual_action = (action + 1) % 4
        else:
            actual_action = (action - 1) % 4

        self.state = self.task._move(self.state, actual_action)
        self.step_count += 1

        reward = 1.0 if self.state == self.task.goal_state else 0.0
        done = self.state in self.task.holes or self.state == self.task.goal_state or self.step_count >= self.max_steps
        return self.task._one_hot(self.state), reward, done

    def close(self):
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDQN baselines and evaluate them under non-stationary settings.")
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
    parser.add_argument("--cartpole-train-steps", type=int, default=300000, help="Environment-step training budget for CartPole.")
    parser.add_argument("--frozenlake-train-steps", type=int, default=1000000, help="Environment-step training budget for FrozenLake.")
    parser.add_argument("--cartpole-max-steps", type=int, default=2500, help="CartPole step cap per episode.")
    parser.add_argument("--frozenlake-max-steps", type=int, default=200, help="FrozenLake step cap per episode.")
    parser.add_argument("--batch-size", type=int, default=64, help="Replay batch size.")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer capacity.")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Replay warmup before updates.")
    parser.add_argument("--train-frequency", type=int, default=1, help="Gradient updates every N env steps.")
    parser.add_argument("--target-update-frequency", type=int, default=1000, help="Target sync frequency in env steps.")
    parser.add_argument("--cartpole-gamma", type=float, default=0.99, help="Discount factor for CartPole.")
    parser.add_argument("--frozenlake-gamma", type=float, default=0.95, help="Discount factor for FrozenLake.")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Baseline evaluation episodes per checkpoint/final run.")
    parser.add_argument("--shift-eval-episodes", type=int, default=20, help="Episodes per shifted-environment evaluation.")
    parser.add_argument("--eval-every", type=int, default=25, help="Evaluate on the baseline every N training episodes.")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon.")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Final epsilon.")
    parser.add_argument("--epsilon-decay-steps", type=int, default=20000, help="Linear epsilon decay horizon in env steps.")
    parser.add_argument(
        "--exploration-policy",
        default="boltzmann",
        choices=["boltzmann", "epsilon_greedy"],
        help="Action-selection policy used during DDQN training.",
    )
    parser.add_argument("--boltzmann-temperature", type=float, default=1.0, help="Softmax temperature for Boltzmann exploration.")
    parser.add_argument("--shared-lr", type=float, default=None, help="Override learning rate for all optimizers.")
    parser.add_argument("--sgd-lr", type=float, default=0.01, help="Learning rate for plain SGD.")
    parser.add_argument("--sgd-momentum-lr", type=float, default=0.01, help="Learning rate for SGD with momentum.")
    parser.add_argument("--sgd-nag-lr", type=float, default=0.01, help="Learning rate for SGD with Nesterov momentum.")
    parser.add_argument("--adam-lr", type=float, default=0.001, help="Learning rate for Adam.")
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="Beta1 (first-moment momentum) for Adam.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="Beta2 (second-moment decay) for Adam.")
    parser.add_argument("--adamw-lr", type=float, default=0.001, help="Learning rate for AdamW.")
    parser.add_argument("--adamw-beta1", type=float, default=0.9, help="Beta1 (first-moment momentum) for AdamW.")
    parser.add_argument("--adamw-beta2", type=float, default=0.999, help="Beta2 (second-moment decay) for AdamW.")
    parser.add_argument("--rmsprop-lr", type=float, default=0.001, help="Learning rate for RMSprop.")
    parser.add_argument("--muon-lr", type=float, default=0.001, help="Learning rate for Muon.")
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
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--cartpole-gravity-settings", nargs="+", type=float, default=[9.8, 20.0, 50.0, 500.0], help="CartPole gravity shifts.")
    parser.add_argument("--cartpole-masscart-settings", nargs="+", type=float, default=[0.1, 1.0, 1.2, 1.3, 1.5], help="CartPole masscart shifts.")
    parser.add_argument("--output-dir", default="results/ddqn_nonstationary_sweep", help="Sweep output directory.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    return parser.parse_args()


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
    if name == "adam":
        lr = args.shared_lr if args.shared_lr is not None else args.adam_lr
        return {
            "name": name,
            "lr": lr,
            "weight_decay": args.weight_decay,
            "beta1": args.adam_beta1,
            "beta2": args.adam_beta2,
        }
    if name == "adamw":
        lr = args.shared_lr if args.shared_lr is not None else args.adamw_lr
        return {
            "name": name,
            "lr": lr,
            "weight_decay": args.weight_decay,
            "beta1": args.adamw_beta1,
            "beta2": args.adamw_beta2,
        }
    if name == "rmsprop":
        lr = args.shared_lr if args.shared_lr is not None else args.rmsprop_lr
        return {"name": name, "lr": lr}
    if name == "muon":
        lr = args.shared_lr if args.shared_lr is not None else args.muon_lr
        fallback_lr = args.shared_lr if args.shared_lr is not None else args.muon_adamw_lr
        if fallback_lr is None:
            fallback_lr = lr
        return {
            "name": name,
            "lr": lr,
            "momentum": args.muon_momentum,
            "ns_steps": args.muon_ns_steps,
            "weight_decay": args.muon_weight_decay,
            "fallback_lr": fallback_lr,
            "fallback_weight_decay": args.muon_adamw_weight_decay,
        }
    raise ValueError(f"Unsupported optimizer: {name}")


def split_muon_param_groups(model: nn.Module):
    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    matrix_param_names = [name for name, param in named_params if param.ndim >= 2]

    input_output_matrix_names = set()
    if matrix_param_names:
        input_output_matrix_names.add(matrix_param_names[0])
        input_output_matrix_names.add(matrix_param_names[-1])

    muon_named_params = []
    fallback_params = []
    for name, param in named_params:
        if param.ndim >= 2 and name not in input_output_matrix_names:
            muon_named_params.append((name, param))
        else:
            fallback_params.append(param)

    return muon_named_params, fallback_params


def build_optimizer(
    parameters,
    spec,
    model=None,
    muon_spectrum_logger=None,
    muon_spectrum_every=0,
    muon_spectrum_topk=0,
    adamw_momentum_logger=None,
    adamw_momentum_every=0,
    sgd_momentum_logger=None,
    sgd_momentum_every=0,
):
    if spec["name"] == "adam":
        adam_opt = torch.optim.Adam(
            parameters,
            lr=spec["lr"],
            weight_decay=spec["weight_decay"],
            betas=(spec["beta1"], spec["beta2"]),
        )
        if adamw_momentum_logger is None or adamw_momentum_every <= 0:
            return adam_opt
        if model is None:
            raise ValueError("Adam momentum logging requires model to map parameter names")
        adam_name_map = {id(param): name for name, param in model.named_parameters() if param.requires_grad}
        return HybridOptimizer(
            adam_opt,
            adamw_momentum_logger=adamw_momentum_logger,
            adamw_momentum_every=adamw_momentum_every,
            adamw_param_name_map=adam_name_map,
        )
    if spec["name"] == "adamw":
        adamw_opt = torch.optim.AdamW(
            parameters,
            lr=spec["lr"],
            weight_decay=spec["weight_decay"],
            betas=(spec["beta1"], spec["beta2"]),
        )
        if adamw_momentum_logger is None or adamw_momentum_every <= 0:
            return adamw_opt
        if model is None:
            raise ValueError("AdamW momentum logging requires model to map parameter names")
        adamw_name_map = {id(param): name for name, param in model.named_parameters() if param.requires_grad}
        return HybridOptimizer(
            adamw_opt,
            adamw_momentum_logger=adamw_momentum_logger,
            adamw_momentum_every=adamw_momentum_every,
            adamw_param_name_map=adamw_name_map,
        )
    if spec["name"] == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=spec["lr"])
    if spec["name"] == "muon":
        if model is None:
            raise ValueError("Muon optimizer requires a model for parameter partitioning")
        muon_named_params, fallback_params = split_muon_param_groups(model)
        muon_params = [param for _name, param in muon_named_params]
        muon_name_map = {id(param): name for name, param in muon_named_params}
        fallback_ids = {id(param) for param in fallback_params}
        fallback_name_map = {id(param): name for name, param in model.named_parameters() if id(param) in fallback_ids}
        muon_opt = (
            Muon(
                muon_params,
                lr=spec["lr"],
                momentum=spec["momentum"],
                ns_steps=spec["ns_steps"],
                weight_decay=spec["weight_decay"],
                param_name_map=muon_name_map,
                spectrum_logger=muon_spectrum_logger,
                spectrum_every=muon_spectrum_every,
                spectrum_topk=muon_spectrum_topk,
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
        return HybridOptimizer(
            muon_opt,
            fallback_opt,
            adamw_momentum_logger=adamw_momentum_logger,
            adamw_momentum_every=adamw_momentum_every,
            adamw_param_name_map=fallback_name_map,
        )
    sgd_opt = torch.optim.SGD(
        parameters,
        lr=spec["lr"],
        momentum=spec.get("momentum", 0.0),
        nesterov=spec.get("nesterov", False),
    )
    if sgd_momentum_logger is None or sgd_momentum_every <= 0 or spec.get("momentum", 0.0) <= 0.0:
        return sgd_opt
    if model is None:
        raise ValueError("SGD momentum logging requires model to map parameter names")
    sgd_name_map = {id(param): name for name, param in model.named_parameters() if param.requires_grad}
    return HybridOptimizer(
        sgd_opt,
        sgd_momentum_logger=sgd_momentum_logger,
        sgd_momentum_every=sgd_momentum_every,
        sgd_param_name_map=sgd_name_map,
    )


def linear_epsilon(step_count, start, end, decay_steps):
    if decay_steps <= 0:
        return end
    frac = min(1.0, step_count / decay_steps)
    return start + frac * (end - start)


def select_action(policy_net, state, epsilon, action_dim, device, exploration_policy, boltzmann_temperature):
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = policy_net(state_tensor).squeeze(0)

    if exploration_policy == "epsilon_greedy":
        if random.random() < epsilon:
            return random.randrange(action_dim)
        return int(torch.argmax(q_values).item())

    scaled_q = q_values / max(boltzmann_temperature, 1e-6)
    probs = torch.softmax(scaled_q, dim=0).cpu().numpy()
    return int(np.random.choice(action_dim, p=probs))


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


def evaluate_on_task(policy_net, task, env_kwargs, episodes, seed, device):
    rewards = []
    max_steps = task.max_steps
    for episode in range(episodes):
        env_seed = seed + 10_000 + episode
        env = task.make_env(env_seed, **env_kwargs)
        state = task.reset(env, env_seed)
        episode_reward = 0.0
        for _ in range(max_steps):
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(policy_net(state_tensor).argmax(dim=1).item())
            state, reward, done = task.step(env, action)
            episode_reward += reward
            if done:
                break
        rewards.append(episode_reward)
        env.close()
    return rewards


def get_task(name: str, args):
    if name == "cartpole":
        return CartPoleTask(max_steps=args.cartpole_max_steps)
    if name == "frozenlake":
        return FrozenLake3x3Task(max_steps=args.frozenlake_max_steps)
    raise ValueError(f"Unsupported domain: {name}")


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


def get_shift_settings(name: str, task, args):
    if name == "cartpole":
        return task.shift_settings(args.cartpole_gravity_settings, args.cartpole_masscart_settings)
    if name == "frozenlake":
        return task.shift_settings()
    raise ValueError(f"Unsupported domain: {name}")


def build_shift_env_kwargs(domain_name: str, shift_setting):
    if domain_name == "cartpole":
        return {
            "gravity": shift_setting.get("gravity"),
            "masscart": shift_setting.get("masscart"),
        }
    if domain_name == "frozenlake":
        return {"prob_distribution": shift_setting["prob_distribution"]}
    raise ValueError(f"Unsupported domain: {domain_name}")


def build_baseline_env_kwargs(domain_name: str):
    if domain_name == "cartpole":
        return {"gravity": None, "masscart": None}
    if domain_name == "frozenlake":
        return {"prob_distribution": FROZENLAKE_BASELINE}
    raise ValueError(f"Unsupported domain: {domain_name}")


def run_single_experiment(domain_name: str, optimizer_name: str, seed: int, args, run_dir: Path, device: torch.device):
    task = get_task(domain_name, args)
    hyperparams = get_domain_hyperparams(domain_name, args)
    episode_cap = hyperparams["episode_cap"]
    train_steps = hyperparams["train_steps"]
    gamma = hyperparams["gamma"]
    set_all_seeds(seed)

    env = task.make_env(seed, **build_baseline_env_kwargs(domain_name))
    policy_net = QNetwork(task.state_dim, task.action_dim).to(device)
    target_net = QNetwork(task.state_dim, task.action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    spec = optimizer_spec(optimizer_name, args)
    spectrum_logger = None
    adamw_momentum_logger = None
    sgd_momentum_logger = None
    if optimizer_name == "muon":
        adamw_momentum_every = args.adamw_momentum_every if args.adamw_momentum_every > 0 else args.muon_adamw_momentum_every
    else:
        adamw_momentum_every = args.adamw_momentum_every
    if optimizer_name == "muon" and args.muon_spectrum_every > 0:
        spectrum_path = run_dir / "diagnostics" / f"{domain_name}_{optimizer_name}_seed{seed}_muon_update_spectrum.csv"
        spectrum_logger = MuonSpectrumCSVLogger(spectrum_path, domain_name, optimizer_name, seed)
    if optimizer_name in {"muon", "adamw", "adam"} and adamw_momentum_every > 0:
        momentum_path = run_dir / "diagnostics" / f"{domain_name}_{optimizer_name}_seed{seed}_adamw_momentum.csv"
        adamw_momentum_logger = AdamWMomentumCSVLogger(momentum_path, domain_name, optimizer_name, seed)
    if optimizer_name in {"sgd_momentum", "sgd_nag"} and args.sgd_momentum_log_every > 0:
        sgd_momentum_path = run_dir / "diagnostics" / f"{domain_name}_{optimizer_name}_seed{seed}_sgd_momentum.csv"
        sgd_momentum_logger = SGDMomentumCSVLogger(sgd_momentum_path, domain_name, optimizer_name, seed)

    optimizer = build_optimizer(
        policy_net.parameters(),
        spec,
        model=policy_net,
        muon_spectrum_logger=spectrum_logger,
        muon_spectrum_every=args.muon_spectrum_every,
        muon_spectrum_topk=args.muon_spectrum_topk,
        adamw_momentum_logger=adamw_momentum_logger,
        adamw_momentum_every=adamw_momentum_every,
        sgd_momentum_logger=sgd_momentum_logger,
        sgd_momentum_every=args.sgd_momentum_log_every,
    )
    replay_buffer = ReplayBuffer(args.buffer_size)

    history_rows = []
    eval_rows = []
    shift_rows = []
    total_steps = 0
    start_time = time.time()

    print(
        f"[start] domain={domain_name} optimizer={optimizer_name} seed={seed} "
        f"train_steps={train_steps} episode_cap={episode_cap}",
        flush=True,
    )

    episode = 0
    while total_steps < train_steps and (episode_cap <= 0 or episode < episode_cap):
        episode += 1
        state = task.reset(env, seed + episode)
        episode_reward = 0.0
        episode_losses = []

        for _ in range(task.max_steps):
            epsilon = linear_epsilon(total_steps, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
            action = select_action(
                policy_net,
                state,
                epsilon,
                task.action_dim,
                device,
                args.exploration_policy,
                args.boltzmann_temperature,
            )
            next_state, reward, done = task.step(env, action)
            replay_buffer.append(Transition(state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward
            total_steps += 1

            if total_steps >= args.warmup_steps and total_steps % args.train_frequency == 0:
                loss = optimize_model(policy_net, target_net, optimizer, replay_buffer, args.batch_size, gamma, device)
                if loss is not None:
                    episode_losses.append(loss)

            if total_steps % args.target_update_frequency == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done or total_steps >= train_steps:
                break

        history_rows.append(
            {
                "domain": domain_name,
                "optimizer": optimizer_name,
                "seed": seed,
                "episode": episode,
                "episode_reward": episode_reward,
                "epsilon": linear_epsilon(total_steps, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps),
                "total_steps": total_steps,
                "mean_loss": float(np.mean(episode_losses)) if episode_losses else None,
            }
        )

        if episode % args.eval_every == 0 or total_steps >= train_steps:
            baseline_rewards = evaluate_on_task(
                policy_net,
                task,
                build_baseline_env_kwargs(domain_name),
                args.eval_episodes,
                seed + episode,
                device,
            )
            eval_rows.append(
                {
                    "domain": domain_name,
                    "optimizer": optimizer_name,
                    "seed": seed,
                    "episode": episode,
                    "eval_mean_reward": float(np.mean(baseline_rewards)),
                    "eval_std_reward": float(np.std(baseline_rewards)),
                }
            )
            print(
                f"[progress] domain={domain_name} optimizer={optimizer_name} seed={seed} "
                f"episode={episode} steps={total_steps}/{train_steps} train_reward={episode_reward:.3f} "
                f"baseline_eval_mean={np.mean(baseline_rewards):.3f}",
                flush=True,
            )

    print(
        f"[shift-eval] domain={domain_name} optimizer={optimizer_name} seed={seed} starting shifted evaluations",
        flush=True,
    )
    for shift_setting in get_shift_settings(domain_name, task, args):
        rewards = evaluate_on_task(
            policy_net,
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
            f"{shift_setting['shift_family']}={shift_setting['shift_value']} "
            f"mean_reward={np.mean(rewards):.3f}",
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
    torch.save(policy_net.state_dict(), model_path)
    history_df.to_csv(history_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    shift_df.to_csv(shift_path, index=False)

    result = {
        "domain": domain_name,
        "optimizer": optimizer_name,
        "seed": seed,
        "optimizer_config": json.dumps(spec, sort_keys=True),
        "episodes": episode,
        "episode_cap": episode_cap,
        "train_steps_budget": train_steps,
        "max_steps": task.max_steps,
        "total_steps": total_steps,
        "elapsed_seconds": time.time() - start_time,
        "last_20_train_reward_mean": float(history_df["episode_reward"].tail(20).mean()),
        "best_train_reward": float(history_df["episode_reward"].max()),
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
    if spectrum_logger is not None:
        spectrum_logger.close()
    if adamw_momentum_logger is not None:
        adamw_momentum_logger.close()
    if sgd_momentum_logger is not None:
        sgd_momentum_logger.close()
    return result


def build_training_summary(raw_results: pd.DataFrame):
    return (
        raw_results.groupby(["domain", "optimizer"], as_index=False)
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
        .sort_values(["domain", "final_eval_mean_reward_mean"], ascending=[True, False])
    )


def build_shift_summary(all_shift_evals: pd.DataFrame):
    return (
        all_shift_evals.groupby(["domain", "shift_family", "shift_value", "optimizer"], as_index=False)
        .agg(
            num_runs=("seed", "count"),
            mean_reward_mean=("mean_reward", "mean"),
            mean_reward_std=("mean_reward", "std"),
            mean_within_run_std=("std_reward", "mean"),
        )
        .sort_values(["domain", "shift_family", "shift_value", "mean_reward_mean"], ascending=[True, True, True, False])
    )


def write_manifest(run_dir: Path, args):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "device": args.device,
        "args": vars(args),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _run_plot_command(cmd):
    print(f"[plot] running: {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        print(f"[plot] failed (exit={completed.returncode}): {' '.join(cmd)}", flush=True)
        if completed.stdout:
            print(completed.stdout.strip(), flush=True)
        if completed.stderr:
            print(completed.stderr.strip(), flush=True)
        return False
    if completed.stdout:
        print(completed.stdout.strip(), flush=True)
    return True


def generate_auto_plots(run_dir: Path, args):
    script_dir = Path(__file__).resolve().parent
    python_exe = sys.executable
    plots_dir = run_dir / "plots"

    comparison_script = script_dir / "plot_ddqn_comparison.py"
    if comparison_script.exists():
        _run_plot_command(
            [
                python_exe,
                str(comparison_script),
                str(run_dir),
                "--output-dir",
                str(plots_dir / "comparison"),
                "--step-bin",
                str(max(0, int(args.auto_plot_step_bin))),
            ]
        )
    else:
        print(f"[plot] skipped comparison (missing {comparison_script.name})", flush=True)

    diagnostics_dir = run_dir / "diagnostics"
    has_muon_spectrum = diagnostics_dir.exists() and any(diagnostics_dir.glob("*_muon_update_spectrum.csv"))
    has_adamw_momentum = diagnostics_dir.exists() and any(diagnostics_dir.glob("*_adamw_momentum.csv"))

    muon_script = script_dir / "plot_muon_update_spectra.py"
    if has_muon_spectrum and muon_script.exists():
        _run_plot_command(
            [
                python_exe,
                str(muon_script),
                str(run_dir),
                "--max-singular-indices",
                "0",
                "--output-dir",
                str(plots_dir / "muon_spectrum"),
            ]
        )
    elif has_muon_spectrum:
        print(f"[plot] skipped Muon spectrum (missing {muon_script.name})", flush=True)

    adamw_momentum_script = script_dir / "plot_adamw_momentum.py"
    if has_adamw_momentum and adamw_momentum_script.exists():
        _run_plot_command(
            [
                python_exe,
                str(adamw_momentum_script),
                str(run_dir),
                "--output-dir",
                str(plots_dir / "adamw_momentum"),
            ]
        )
    elif has_adamw_momentum:
        print(f"[plot] skipped AdamW momentum (missing {adamw_momentum_script.name})", flush=True)


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
