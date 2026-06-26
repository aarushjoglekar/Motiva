from music.song import Song
from environment.environment import Environment
from environment.episodetype import EpisodeType
from ml.sac_droq import SAC_DROQ
from ml.config import SAC_DROQ_DEFAULT_CONFIG
import numpy as np
import time
import os

### SETTINGS
# GENERAL SETTINGS
MODEL_NAME = "motiva"

# TRAINING SETTINGS
TRAINING = False
NUM_EPISODES = 3
VALIDATION_INTERVAL = 2

# TESTING SETTINGS
SAVE_TO_MIDI = False

# SONG SETTINGS
SONG = Song.from_txt(Song.SOMEWHERE_OVER_THE_RAINBOW)


def run_episode(
    model: SAC_DROQ,
    env: Environment,
    episode_type: EpisodeType,
    model_path: str,
    on_step_end=None,
):
    observation = env.reset(
        play_audio=(episode_type == EpisodeType.TEST),
        record_midi=(
            episode_type == EpisodeType.VALIDATION
            or (episode_type == EpisodeType.TEST and SAVE_TO_MIDI)
        ),
        midi_path=model_path,
    )

    if episode_type == EpisodeType.TRAINING:
        model.train()
    else:
        model.eval()

    while True:
        observation, reward, done = env.step(action=np.zeros(env.num_actions()))

        if on_step_end is not None:
            on_step_end()

        if done:
            return False

        if episode_type == EpisodeType.TEST and not env.viewer_running():
            return True


def run_test(model: SAC_DROQ, env: Environment, model_path: str):
    time.sleep(0.5)

    STEP_DURATION = 1 / Song.RESOLUTION

    def pace():
        sleep_time = (
            (env.start_time + 1e9 * env.step_count * STEP_DURATION)
            - time.perf_counter_ns()
        ) / 1e9

        if sleep_time > 0:
            time.sleep(sleep_time)

    closed_viewer = run_episode(
        model=model,
        env=env,
        episode_type=EpisodeType.TEST,
        model_path=model_path,
        on_step_end=pace,
    )
    if not closed_viewer:
        time.sleep(2)


def run_training(model: SAC_DROQ, env: Environment, model_path: str):
    os.makedirs(model_path, exist_ok=True)
    for episode in range(NUM_EPISODES):
        run_episode(
            model=model,
            env=env,
            episode_type=(
                EpisodeType.VALIDATION
                if ((episode + 1) % VALIDATION_INTERVAL == 0)
                else EpisodeType.TRAINING
            ),
            model_path=model_path,
        )


with Environment(SONG, should_render=(not TRAINING)) as env:
    DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(DIR, f"ml/models/{MODEL_NAME}")

    model = SAC_DROQ(
        model_path=model_path,
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
        target_entropy=SAC_DROQ_DEFAULT_CONFIG.target_entropy,
        discount_factor=SAC_DROQ_DEFAULT_CONFIG.discount_factor,
        tau=SAC_DROQ_DEFAULT_CONFIG.tau,
    )

    if TRAINING:
        run_training(model=model, env=env, model_path=model_path)
    else:
        run_test(model=model, env=env, model_path=model_path)
