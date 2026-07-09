from ml.replay_buffer import ReplayBuffer
from ml.actor import Actor
from ml.critic import BatchedCritic
import torch
import os
import math


class SAC_DROQ(torch.nn.Module):
    def __init__(
        self,
        model_path: str,
        num_observations: int,
        num_actions: int,
        actor_hidden_layer_size: int,
        actor_num_hidden_layers: int,
        critic_hidden_layer_size: int,
        critic_num_hidden_layers: int,
        num_critics: int,
        actor_lr: float,
        critic_lr: float,
        log_alpha_lr: float,
        critic_dropout_probability: float,
        min_action_log_std: float,
        max_action_log_std: float,
        warmup_samples: int,
        updates_per_step: int,
        sample_size: int,
        replay_buffer_size: int,
        target_entropy: float,
        discount_factor: float,
        tau: float,
        device: str,
    ):
        super().__init__()
        
        self.device = device

        self.actor = Actor(
            hidden_layer_size=actor_hidden_layer_size,
            num_hidden_layers=actor_num_hidden_layers,
            num_observations=num_observations,
            num_actions=num_actions,
        )
        self.actor_optimizer = torch.optim.Adam(
            params=self.actor.parameters(), lr=actor_lr, fused=(device == "cuda")
        )

        self.num_critics = num_critics
        self.critics = BatchedCritic(
            num_critics=num_critics,
            hidden_layer_size=critic_hidden_layer_size,
            num_hidden_layers=critic_num_hidden_layers,
            num_observations=num_observations,
            num_actions=num_actions,
            dropout_probability=critic_dropout_probability,
        )
        self.target_critics = BatchedCritic(
            num_critics=num_critics,
            hidden_layer_size=critic_hidden_layer_size,
            num_hidden_layers=critic_num_hidden_layers,
            num_observations=num_observations,
            num_actions=num_actions,
            dropout_probability=critic_dropout_probability,
        )

        self.target_critics.load_state_dict(self.critics.state_dict())

        self.critic_params = list(self.critics.parameters())
        self.critic_target_params = list(self.target_critics.parameters())

        self.critic_optimizer = torch.optim.Adam(self.critic_params, lr=critic_lr, fused=(device == "cuda"))

        self.target_entropy = target_entropy
        self.log_alpha = torch.nn.Parameter(torch.zeros(1))
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=log_alpha_lr, fused=(device == "cuda"))

        self.min_action_log_std = min_action_log_std
        self.max_action_log_std = max_action_log_std
        self.warmup_samples = warmup_samples
        self.updates_per_step = updates_per_step

        self.discount_factor = discount_factor
        self.tau = tau

        self.replay_buffer = ReplayBuffer(
            num_observations=num_observations,
            num_actions=num_actions,
            sample_size=sample_size,
            max_size=replay_buffer_size,
            device=device,
        )

        def optimizer_to_device(optimizer: torch.optim.Optimizer, device: str):
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        self.model_path = model_path
        try:
            loaded = torch.load(
                os.path.join(self.model_path, "model.pth"),
                weights_only=True,
                map_location=device,
            )
            self.load_state_dict(loaded["weights"])
            self.actor_optimizer.load_state_dict(loaded["actor_optimizer"])
            self.critic_optimizer.load_state_dict(loaded["critic_optimizer"])
            self.log_alpha_optimizer.load_state_dict(loaded["log_alpha_optimizer"])

            optimizer_to_device(optimizer=self.actor_optimizer, device=device)
            optimizer_to_device(optimizer=self.critic_optimizer, device=device)
            optimizer_to_device(optimizer=self.log_alpha_optimizer, device=device)
        except FileNotFoundError:
            print("Model not loaded: instantiating new model")

        try:
            loaded = torch.load(
                os.path.join(self.model_path, "replay_buffer.pth"),
                weights_only=True,
                map_location=device,
            )
            self.replay_buffer.load(loaded["replay_buffer"])
        except FileNotFoundError:
            print("Replay buffer not loaded")

    def select_action(self, state: torch.Tensor, deterministic: bool):
        y = self.actor(state)

        means, log_stds = y.chunk(2, dim=-1)

        if deterministic:
            return torch.tanh(means), torch.tensor([])

        clamped_log_stds = torch.clamp(
            input=log_stds, min=self.min_action_log_std, max=self.max_action_log_std
        )

        dist = torch.distributions.Normal(means, clamped_log_stds.exp())

        action = dist.rsample()
        log_prob = dist.log_prob(action)

        scaled_action = torch.tanh(action)
        scaled_log_prob = (
            log_prob
            - 2 * (math.log(2) - action - torch.nn.functional.softplus(-2 * action))
        ).sum(-1)

        return scaled_action, scaled_log_prob

    def update(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
    ):
        self.replay_buffer.add_sample(
            state=state.detach(),
            next_state=next_state.detach(),
            action=action.detach(),
            reward=reward,
        )

        if (
            self.replay_buffer.has_enough_samples()
            and self.replay_buffer.length > self.warmup_samples
        ):

            avg_critic_loss = torch.zeros((), device=self.device)

            for _ in range(self.updates_per_step):
                states, actions, rewards, next_states = (
                    self.replay_buffer.sample_random()
                )

                with torch.no_grad():
                    next_actions, next_log_probs = self.select_action(
                        state=next_states, deterministic=False
                    )
                    next_q = torch.min(
                        self.target_critics.forward(
                            state=next_states, action=next_actions, dropout=True
                        ),
                        dim=0,
                    ).values
                    critic_target = rewards + self.discount_factor * (
                        next_q - self.log_alpha.exp() * next_log_probs
                    )

                self.critic_optimizer.zero_grad()
                critic_losses = (
                    (
                        self.critics.forward(state=states, action=actions, dropout=True)
                        - critic_target
                    )
                    ** 2
                ).mean(dim=1)
                critic_loss = critic_losses.sum()
                critic_loss.backward()
                self.critic_optimizer.step()

                avg_critic_loss += critic_losses.mean().detach()

                with torch.no_grad():
                    torch._foreach_lerp_(self.critic_target_params, self.critic_params, self.tau)  # type: ignore

            current_actions, current_log_probs = self.select_action(
                state=states, deterministic=False
            )

            self.actor_optimizer.zero_grad()
            actor_loss = (
                -self.critics.forward(
                    state=states, action=current_actions, dropout=False
                ).mean(dim=0)
                + self.log_alpha.exp() * current_log_probs
            ).mean()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.log_alpha_optimizer.zero_grad()
            log_alpha_loss = -(
                self.log_alpha.exp()
                * (current_log_probs + self.target_entropy).detach()
            ).mean()
            log_alpha_loss.backward()
            self.log_alpha_optimizer.step()

            return (
                actor_loss.detach(),
                (avg_critic_loss / self.updates_per_step).detach(),
                current_log_probs.mean().detach(),
                self.log_alpha.exp().squeeze(0).detach(),
            )

        return None

    def save(self):
        torch.save(
            {
                "weights": self.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "log_alpha_optimizer": self.log_alpha_optimizer.state_dict(),
            },
            os.path.join(self.model_path, "model.pth"),
        )

        print("Model Saved!")

        torch.save(
            {"replay_buffer": self.replay_buffer.dump()},
            os.path.join(self.model_path, "replay_buffer.pth"),
        )

        print("Replay Buffer Saved!")
