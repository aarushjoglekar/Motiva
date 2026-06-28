import torch
import math
from ml.replay_buffer import ReplayBuffer
import os


def init_weights_xavier(module: torch.nn.Module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        torch.nn.init.zeros_(module.bias)


class Actor(torch.nn.Module):
    def __init__(
        self,
        hidden_layer_size: int,
        num_hidden_layers: int,
        num_observations: int,
        num_actions: int,
    ):
        super().__init__()

        layers = [torch.nn.Linear(num_observations, hidden_layer_size), torch.nn.GELU()]

        for _ in range(num_hidden_layers):
            layers.append(torch.nn.Linear(hidden_layer_size, hidden_layer_size))
            layers.append(torch.nn.GELU())

        layers.append(torch.nn.Linear(hidden_layer_size, num_actions * 2))

        self.layers = torch.nn.Sequential(*layers)

        self.apply(init_weights_xavier)

    def forward(self, state: torch.Tensor):
        return self.layers(state)


class Critic(torch.nn.Module):
    def __init__(
        self,
        hidden_layer_size: int,
        num_hidden_layers: int,
        num_observations: int,
        num_actions: int,
        dropout_probability: float,
    ):
        super().__init__()

        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(num_observations + num_actions, hidden_layer_size),
            torch.nn.LayerNorm(hidden_layer_size),
        )

        self.hidden_layers = torch.nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_layer_size, hidden_layer_size),
                    torch.nn.LayerNorm(hidden_layer_size),
                )
            )

        self.output_layer = torch.nn.Linear(hidden_layer_size, 1)

        self.dropout = torch.nn.Dropout(p=dropout_probability)
        self.gelu = torch.nn.GELU()

        self.apply(init_weights_xavier)

    def forward(self, state: torch.Tensor, action: torch.Tensor, dropout: bool):
        X = torch.cat([state, action], dim=-1)

        X = self.gelu(self.apply_dropout(X=self.input_layer(X), dropout=dropout))

        for hidden_layer in self.hidden_layers:
            X = self.gelu(self.apply_dropout(X=hidden_layer(X), dropout=dropout))

        return self.output_layer(X).squeeze(-1)

    def apply_dropout(self, X: torch.Tensor, dropout: bool):
        if dropout:
            X = self.dropout(X)

        return X


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
        device: str
    ):
        super().__init__()

        self.actor = Actor(
            hidden_layer_size=actor_hidden_layer_size,
            num_hidden_layers=actor_num_hidden_layers,
            num_observations=num_observations,
            num_actions=num_actions,
        )
        self.actor_optimizer = torch.optim.Adam(
            params=self.actor.parameters(), lr=actor_lr
        )

        self.num_critics = num_critics
        self.critics = torch.nn.ModuleList()
        self.critic_targets = torch.nn.ModuleList()
        for _ in range(num_critics):
            critic, target = SAC_DROQ.initialize_critic(
                hidden_layer_size=critic_hidden_layer_size,
                num_hidden_layers=critic_num_hidden_layers,
                num_observations=num_observations,
                num_actions=num_actions,
                dropout_probability=critic_dropout_probability,
            )
            self.critics.append(critic)
            self.critic_targets.append(target)

        self.critic_params = [
            param for critic in self.critics for param in critic.parameters()
        ]
        self.critic_target_params = [
            target_param
            for critic_target in self.critic_targets
            for target_param in critic_target.parameters()
        ]

        self.critic_optimizer = torch.optim.Adam(self.critic_params, lr=critic_lr)

        self.target_entropy = target_entropy
        self.log_alpha = torch.nn.Parameter(torch.zeros(1))
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=log_alpha_lr)
        self.alpha = self.log_alpha.exp().item()

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
            device=device
        )

        self.model_path = model_path
        try:
            loaded = torch.load(
                os.path.join(self.model_path, "model.pth"), weights_only=True
            )
            self.load_state_dict(loaded["weights"])

            self.actor_optimizer.load_state_dict(loaded["actor_optimizer"])

            self.critic_optimizer.load_state_dict(loaded["critic_optimizer"])

            self.log_alpha_optimizer.load_state_dict(loaded["log_alpha_optimizer"])

            self.alpha = self.log_alpha.exp().item()
        except FileNotFoundError:
            print("Model not loaded: instantiating new model")

        try:
            loaded = torch.load(
                os.path.join(self.model_path, "replay_buffer.pth"), weights_only=True
            )
            self.replay_buffer.load(loaded["replay_buffer"])
        except FileNotFoundError:
            print("Replay buffer not loaded")

        # self.actor = torch.compile(self.actor)
        # self.critics = torch.nn.ModuleList([torch.compile(c) for c in self.critics])
        # self.critic_targets = torch.nn.ModuleList([torch.compile(c) for c in self.critic_targets])

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
        truncated: bool,
    ):
        self.replay_buffer.add_sample(
            state=state.detach(),
            next_state=next_state.detach(),
            action=action.detach(),
            reward=reward,
            truncated=truncated,
        )

        if (
            self.replay_buffer.has_enough_samples()
            and self.replay_buffer.length > self.warmup_samples
        ):

            avg_critic_loss = 0

            for _ in range(self.updates_per_step):
                states, actions, rewards, next_states = (
                    self.replay_buffer.sample_random()
                )

                with torch.no_grad():
                    next_actions, next_log_probs = self.select_action(
                        state=next_states, deterministic=False
                    )
                    next_q = (
                        self.batch_critics_forward(
                            critics=self.critic_targets,
                            states=next_states,
                            actions=next_actions,
                            dropout=True,
                        )
                        .min(dim=0)
                        .values
                    )
                    critic_target = rewards + self.discount_factor * (
                        next_q - self.alpha * next_log_probs
                    )

                self.critic_optimizer.zero_grad()
                q_all = self.batch_critics_forward(critics=self.critics, states=states, actions=actions, dropout=True)
                critic_losses = ((q_all - critic_target) ** 2).mean(dim=1)
                critic_loss = critic_losses.sum()
                critic_loss.backward()
                self.critic_optimizer.step()

                avg_critic_loss += critic_losses.mean().item()

                with torch.no_grad():
                    torch._foreach_lerp_(self.critic_target_params, self.critic_params, self.tau)  # type: ignore

            current_actions, current_log_probs = self.select_action(
                state=states, deterministic=False
            )

            self.actor_optimizer.zero_grad()
            q_actor = self.batch_critics_forward(critics=self.critics, states=states, actions=current_actions, dropout=False)
            actor_loss = (-q_actor.mean(dim=0) + self.alpha * current_log_probs).mean()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.log_alpha_optimizer.zero_grad()
            log_alpha_loss = -(
                self.log_alpha.exp()
                * (current_log_probs + self.target_entropy).detach()
            ).mean()
            log_alpha_loss.backward()
            self.log_alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

            return (
                actor_loss.item(),
                avg_critic_loss / self.updates_per_step,
                current_log_probs.mean().item(),
                self.alpha,
            )

        return None

    def batch_critics_forward(
        self,
        critics: torch.nn.ModuleList,
        states: torch.Tensor,
        actions: torch.Tensor,
        dropout: bool,
    ):
        params, buffers = torch.func.stack_module_state(list(critics))
        template = critics[0]

        def single_function(p, b):
            return torch.func.functional_call(
                template, (p, b), (states, actions, dropout)
            )

        return torch.func.vmap(single_function, randomness="different")(params, buffers)

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

    @staticmethod
    def initialize_critic(
        hidden_layer_size: int,
        num_hidden_layers: int,
        num_observations: int,
        num_actions: int,
        dropout_probability: float,
    ):
        critic = Critic(
            hidden_layer_size=hidden_layer_size,
            num_hidden_layers=num_hidden_layers,
            num_observations=num_observations,
            num_actions=num_actions,
            dropout_probability=dropout_probability,
        )

        target = Critic(
            hidden_layer_size=hidden_layer_size,
            num_hidden_layers=num_hidden_layers,
            num_observations=num_observations,
            num_actions=num_actions,
            dropout_probability=dropout_probability,
        )

        target.load_state_dict(critic.state_dict())

        return critic, target
