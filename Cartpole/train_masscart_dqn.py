import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam

from rl.agents.dqn import DQNAgent
from rl.callbacks import FileLogger
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

try:
    from gym_compat import make_legacy_env
except ModuleNotFoundError:
    from Cartpole.gym_compat import make_legacy_env


ENV_NAME = "CartPole-v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the CartPole DDQN baseline and save metrics.")
    parser.add_argument("--nb-steps", type=int, default=300000, help="Training steps.")
    parser.add_argument("--test-episodes", type=int, default=40, help="Evaluation episodes after training.")
    parser.add_argument("--seed", type=int, default=2, help="Environment seed.")
    parser.add_argument(
        "--output-dir",
        default="Cartpole/results/ddqn",
        help="Directory where this training run will be written.",
    )
    parser.add_argument(
        "--weights-name",
        default=f"duel_dqn_{ENV_NAME}_weights_2500_computime.h5f",
        help="Base filename for saved weights.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Interval for the keras-rl JSON training log.",
    )
    return parser.parse_args()


def build_agent(env):
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation("relu"))
    model.add(Dense(16))
    model.add(Activation("relu"))
    model.add(Dense(16))
    model.add(Activation("relu"))
    model.add(Dense(nb_actions, activation="linear"))
    print(model.summary())

    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=500,
        target_model_update=1000,
        policy=policy,
        gamma=0.99,
        train_interval=4,
        enable_double_dqn=True,
    )
    dqn.compile(Adam(learning_rate=1e-3), metrics=["mae"])
    return dqn


def write_summary(summary_path, payload):
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    weights_path = run_dir / args.weights_name
    train_log_path = run_dir / "training_log.json"
    test_csv_path = run_dir / "test_rewards.csv"
    summary_path = run_dir / "summary.json"

    start_time = time.time()

    env = make_legacy_env(ENV_NAME)
    env._max_episode_steps = 2500
    env.reward_threshold = 2500
    env.seed(args.seed)

    dqn = build_agent(env)
    dqn.fit(
        env,
        nb_steps=args.nb_steps,
        visualize=False,
        verbose=2,
        callbacks=[FileLogger(str(train_log_path), interval=args.log_interval)],
    )

    dqn.save_weights(str(weights_path), overwrite=True)

    training_elapsed = time.time() - start_time
    test_history = dqn.test(env, nb_episodes=args.test_episodes, visualize=False)
    test_rewards = test_history.history.get("episode_reward", [])

    pd.DataFrame(
        {
            "episode": list(range(1, len(test_rewards) + 1)),
            "reward": test_rewards,
        }
    ).to_csv(test_csv_path, index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "env_name": ENV_NAME,
        "seed": args.seed,
        "nb_steps": args.nb_steps,
        "test_episodes": args.test_episodes,
        "training_elapsed_seconds": training_elapsed,
        "weights_path": str(weights_path),
        "training_log_path": str(train_log_path),
        "test_csv_path": str(test_csv_path),
        "test_reward_mean": float(pd.Series(test_rewards).mean()) if test_rewards else None,
        "test_reward_std": float(pd.Series(test_rewards).std()) if test_rewards else None,
    }
    write_summary(summary_path, summary)

    print(f"Run directory: {run_dir}")
    print(f"Weights: {weights_path}")
    print(f"Training log: {train_log_path}")
    print(f"Test rewards: {test_csv_path}")
    print(f"Summary: {summary_path}")
    print("done")


if __name__ == "__main__":
    main()
