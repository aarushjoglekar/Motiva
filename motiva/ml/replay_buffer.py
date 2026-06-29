import torch


class ReplayBuffer:
    def __init__(
        self,
        num_observations: int,
        num_actions: int,
        sample_size: int,
        max_size: int,
        device: str,
    ):
        self.max_size = max_size
        self.sample_size = sample_size

        self.length = 0
        self.pointer = 0

        self.states = torch.zeros(self.max_size, num_observations, device=device)
        self.actions = torch.zeros(self.max_size, num_actions, device=device)
        self.rewards = torch.zeros(self.max_size, device=device)
        self.next_states = torch.zeros(self.max_size, num_observations, device=device)

        self.device = device

    def load(self, data):
        self.states = data["states"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.next_states = data["next_states"]
        self.length = data["length"]
        self.pointer = data["pointer"]

    def dump(self):
        return {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_states": self.next_states,
            "length": self.length,
            "pointer": self.pointer,
        }

    def add_sample(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
    ):
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = next_state

        self.pointer += 1
        if self.pointer >= self.max_size:
            self.pointer = 0

        if self.length < self.max_size:
            self.length += 1

    def sample_random(self):
        indices = torch.randint(0, self.length, (self.sample_size,), device=self.device)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
        )

    def size(self):
        return self.length

    def has_enough_samples(self):
        return self.length >= self.sample_size
