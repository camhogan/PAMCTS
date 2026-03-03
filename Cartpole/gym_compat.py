import gym


class LegacyGymWrapper(gym.Wrapper):
    """Expose the pre-0.26 Gym API expected by older training code."""

    def reset(self, **kwargs):
        observation, _info = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if truncated:
            info = dict(info)
            info.setdefault("TimeLimit.truncated", True)
        return observation, reward, done, info

    def seed(self, seed=None):
        self.reset(seed=seed)
        self.action_space.seed(seed)
        if hasattr(self.observation_space, "seed"):
            self.observation_space.seed(seed)
        return [seed]


def make_legacy_env(env_name, *args, **kwargs):
    return LegacyGymWrapper(gym.make(env_name, *args, **kwargs))
