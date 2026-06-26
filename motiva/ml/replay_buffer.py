import torch


class ReplayBuffer:
    def __init__(
        self,
        num_observations,
        num_actions,
        sample_size,
        max_size,
    ):
        self.max_size = max_size
        self.sample_size = sample_size

        self.length = 0
        self.pointer = 0

        self.states = torch.zeros(self.max_size, num_observations)
        self.actions = torch.zeros(self.max_size, num_actions)
        self.rewards = torch.zeros(self.max_size, 1)
        self.dones = torch.zeros(self.max_size, 1)

    def load(self, data):
        self.states = data["states"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.dones = data["dones"]
        self.length = data["length"]
        self.pointer = data["pointer"]

    def dump(self):
        return {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "length": self.length,
            "pointer": self.pointer,
        }

    def add_sample(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        done: bool,
    ):
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.dones[self.pointer] = done

        self.pointer += 1
        if self.pointer >= self.max_size:
            self.pointer = 0

        if self.length < self.max_size:
            self.length += 1

    def sample_random(self):
        indices = torch.randint(
            0, self.length - 1, (self.sample_size,)
        )  # sample one less than the lenght

        forbidden = (self.pointer - 1) % self.length
        indices[
            indices >= forbidden
        ] += 1  # shift everything above the pointer by 1 to keep results uniform

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices].squeeze(-1),
            self.states[(indices + 1) % self.length],
            self.dones[indices].squeeze(-1),
        )

    def size(self):
        return self.length

    def has_enough_samples(self):
        return self.length >= self.sample_size
