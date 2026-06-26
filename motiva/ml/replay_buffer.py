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
        self.rewards = torch.zeros(self.max_size)
        self.is_truncated = torch.zeros(self.max_size, dtype=torch.bool)

        self.truncated_next_states = dict()

    def load(self, data):
        self.states = data["states"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.is_truncated = data["is_truncated"]
        self.truncated_next_states = data["truncated_next_states"]
        self.length = data["length"]
        self.pointer = data["pointer"]

    def dump(self):
        return {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "is_truncated": self.is_truncated,
            "truncated_next_states": self.truncated_next_states,
            "length": self.length,
            "pointer": self.pointer,
        }

    def add_sample(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        truncated: bool,
    ):
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.is_truncated[self.pointer] = truncated

        if truncated:
            self.truncated_next_states[self.pointer] = next_state.detach().clone()
        else:
            self.truncated_next_states.pop(self.pointer, None)

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

        next_states = self.states[(indices + 1) % self.length]

        is_truncated = self.is_truncated[indices]
        if is_truncated.any():
            hit_positions = is_truncated.nonzero(as_tuple=True)[
                0
            ]  # which index in the selected batch order
            hit_buffer_indices = indices[
                hit_positions
            ]  # which index in the entire buffer
            replacement = torch.stack(
                [
                    self.truncated_next_states[index.item()]
                    for index in hit_buffer_indices
                ]
            )
            next_states[hit_positions] = replacement

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            next_states,
        )

    def size(self):
        return self.length

    def has_enough_samples(self):
        return self.length >= self.sample_size
