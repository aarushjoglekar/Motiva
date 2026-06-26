from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class SAC_DROQ_DEFAULT_CONFIG:
    actor_hidden_layer_size: int = 0
    actor_num_hidden_layers: int = 0
    critic_hidden_layer_size: int = 0
    critic_num_hidden_layers: int = 0
    num_critics: int = 0
    actor_lr: float = 0
    critic_lr: float = 0
    log_alpha_lr: float = 0
    critic_dropout_probability: float = 0
    min_action_log_std: float = 0
    max_action_log_std: float = 0
    warmup_samples: int = 0
    updates_per_step: int = 0
    sample_size: int = 0
    replay_buffer_size: int = 0
    target_entropy: float = 0
    discount_factor: float = 0
    tau: float = 0
