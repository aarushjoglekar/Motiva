import torch

class ReplayBuffer:
    def __init__(self, states_dimension, actions_dimension, rewards_dimension, next_states_dimension, dones_dimension, sample_size, device, max_size):
        self.max_size = max_size
        self.sample_size = sample_size

        self.length = 0
        self.pointer = 0

        self.states = torch.zeros(self.max_size, states_dimension, device=device)
        self.actions = torch.zeros(self.max_size, actions_dimension, device=device)
        self.rewards = torch.zeros(self.max_size, rewards_dimension, device=device)
        self.next_states = torch.zeros(self.max_size, next_states_dimension, device=device)
        self.dones = torch.zeros(self.max_size, dones_dimension, device=device)

    def load(self, data):
        self.states = data["states"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.next_states = data["next_states"]
        self.dones = data["dones"]
        self.length = data["length"]
        self.pointer = data["pointer"]

    def dump(self):
        return {
            "states" : self.states,
            "actions" : self.actions,
            "rewards" : self.rewards,
            "next_states" : self.next_states,
            "dones" : self.dones,
            "length" : self.length,
            "pointer" : self.pointer
        }

    def add_sample(self, state, action, reward, next_state, done):
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = next_state
        self.dones[self.pointer] = done

        self.pointer += 1
        if self.pointer >= self.max_size:
            self.pointer = 0

        if self.length < self.max_size:
            self.length += 1

    def sample_random(self):
        indices = torch.randint(0, self.length, (self.sample_size, ))

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def size(self):
        return self.length
    
    def has_enough_samples(self):
        return self.length >= self.sample_size