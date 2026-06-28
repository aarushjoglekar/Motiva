import time
import os
import torch
import numpy as np

from music.song import Song
from environment.environment import Environment
from ml.sac_droq import SAC_DROQ
from ml.config import SAC_DROQ_DEFAULT_CONFIG

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

SONG_CHOICE = Song.TWINKLE_TWINKLE_LITTLE_STAR
SONG = Song.from_txt(name=SONG_CHOICE)
WARMUP_UPDATES = 200
N = 500

with Environment(SONG, should_render=False) as env:
    DIR = os.path.dirname(os.path.abspath(__file__))
    model = SAC_DROQ(
        model_path=os.path.join(DIR, "ml/models/_bench_tmp"),
        num_observations=env.num_observations(),
        num_actions=env.num_actions(),
        actor_hidden_layer_size=SAC_DROQ_DEFAULT_CONFIG.actor_hidden_layer_size,
        actor_num_hidden_layers=SAC_DROQ_DEFAULT_CONFIG.actor_num_hidden_layers,
        critic_hidden_layer_size=SAC_DROQ_DEFAULT_CONFIG.critic_hidden_layer_size,
        critic_num_hidden_layers=SAC_DROQ_DEFAULT_CONFIG.critic_num_hidden_layers,
        num_critics=SAC_DROQ_DEFAULT_CONFIG.num_critics,
        actor_lr=SAC_DROQ_DEFAULT_CONFIG.actor_lr,
        critic_lr=SAC_DROQ_DEFAULT_CONFIG.critic_lr,
        log_alpha_lr=SAC_DROQ_DEFAULT_CONFIG.log_alpha_lr,
        critic_dropout_probability=SAC_DROQ_DEFAULT_CONFIG.critic_dropout_probability,
        min_action_log_std=SAC_DROQ_DEFAULT_CONFIG.min_action_log_std,
        max_action_log_std=SAC_DROQ_DEFAULT_CONFIG.max_action_log_std,
        warmup_samples=SAC_DROQ_DEFAULT_CONFIG.warmup_samples,
        updates_per_step=SAC_DROQ_DEFAULT_CONFIG.updates_per_step,
        sample_size=SAC_DROQ_DEFAULT_CONFIG.sample_size,
        replay_buffer_size=SAC_DROQ_DEFAULT_CONFIG.replay_buffer_size,
        target_entropy=(-0.5 * env.num_actions()),
        discount_factor=SAC_DROQ_DEFAULT_CONFIG.discount_factor,
        tau=SAC_DROQ_DEFAULT_CONFIG.tau,
    )

    model.train()
    obs = env.reset(play_audio=False, record_midi=False, midi_file="")
    state = torch.from_numpy(obs).float()

    real_updates = 0
    select_time = env_time = update_time = 0.0
    timed = 0

    while timed < N:
        t0 = time.perf_counter()
        action, _ = model.select_action(state=state, deterministic=False)
        t1 = time.perf_counter()

        next_obs, reward, truncated = env.step(action=action.detach().numpy())
        t2 = time.perf_counter()
        next_state = torch.from_numpy(next_obs).float()

        u0 = time.perf_counter()
        updated = model.update(
            state=state,
            next_state=next_state,
            action=action,
            reward=reward,
            truncated=truncated,
        )
        u1 = time.perf_counter()

        if updated is not None:
            real_updates += 1
            if real_updates > WARMUP_UPDATES:
                select_time += t1 - t0
                env_time += t2 - t1
                update_time += u1 - u0
                timed += 1

        state = next_state
        if truncated:
            obs = env.reset(play_audio=False, record_midi=False, midi_file="")
            state = torch.from_numpy(obs).float()

    print(f"select_action: {select_time / N * 1000:.2f} ms/step")
    print(f"env.step: {env_time   / N * 1000:.2f} ms/step")
    print(f"model.update: {update_time/ N * 1000:.2f} ms/step")
    print(f"sum: {(select_time+env_time+update_time)/N*1000:.2f} ms/step")
