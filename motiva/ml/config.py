from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class SAC_DROQ_DEFAULT_CONFIG:
    actor_hidden_layer_size: int = 256
    actor_num_hidden_layers: int = 3
    critic_hidden_layer_size: int = 256
    critic_num_hidden_layers: int = 3
    num_critics: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    log_alpha_lr: float = 3e-4
    critic_dropout_probability: float = 0.01
    min_action_log_std: float = -20
    max_action_log_std: float = 2
    warmup_samples: int = 5000
    updates_per_step: int = 1
    sample_size: int = 256
    replay_buffer_size: int = 1000000
    discount_factor: float = 0.95
    tau: float = 0.005
