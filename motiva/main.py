from music.song import Song
from environment.environment import Environment
from environment.episodetype import EpisodeType
from ml.sac_droq import SAC_DROQ
from ml.config import SAC_DROQ_DEFAULT_CONFIG
import time
import os
import torch

### SETTINGS
# GENERAL SETTINGS
MODEL_NAME = "motiva"

# TRAINING SETTINGS
TRAINING = True
NUM_STEPS = 50000
VALIDATION_INTERVAL = 25000

# TESTING SETTINGS
SAVE_TO_MIDI = False

# SONG SETTINGS
SONG = Song.from_txt(Song.SOMEWHERE_OVER_THE_RAINBOW)


# returns total reward, avg actor loss, avg critic loss, avg log probs, avg alpha
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
    observation = torch.from_numpy(observation).float()

    if episode_type == EpisodeType.TRAINING:
        model.train()
    else:
        model.eval()

    steps = 0
    sum_reward = 0
    sum_actor_loss = 0
    sum_critic_loss = 0
    sum_log_probs = 0
    sum_alpha = 0

    while True:
        action, _ = model.select_action(
            state=observation, deterministic=(episode_type != EpisodeType.TRAINING)
        )

        next_observation, reward, truncated = env.step(action=action.detach().numpy())
        next_observation = torch.from_numpy(next_observation).float()

        steps += 1
        sum_reward += reward

        if episode_type == EpisodeType.TRAINING:
            actor_loss, critic_loss, log_probs, alpha = model.update(
                state=observation,
                next_state=next_observation,
                action=action,
                reward=reward,
                truncated=truncated,
            )

            sum_actor_loss += actor_loss
            sum_critic_loss += critic_loss
            sum_log_probs += log_probs
            sum_alpha += alpha

        observation = next_observation

        if on_step_end is not None:
            on_step_end()

        if truncated:
            return (
                sum_reward,
                sum_actor_loss / steps,
                sum_critic_loss / steps,
                sum_log_probs / steps,
                sum_alpha / steps,
                steps,
                False,
            )

        if episode_type == EpisodeType.TEST and not env.viewer_running():
            return (
                sum_reward,
                sum_actor_loss / steps,
                sum_critic_loss / steps,
                sum_log_probs / steps,
                sum_alpha / steps,
                steps,
                True,
            )


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

    sum_reward, _, _, _, _, _, closed_viewer = run_episode(
        model=model,
        env=env,
        episode_type=EpisodeType.TEST,
        model_path=model_path,
        on_step_end=pace,
    )

    print(f"Test Episode || Total Reward: {sum_reward}")

    if not closed_viewer:
        time.sleep(2)


def run_training(model: SAC_DROQ, env: Environment, model_path: str):
    os.makedirs(model_path, exist_ok=True)

    num_steps = 0
    episode = 0

    start_time = time.perf_counter()

    while num_steps < NUM_STEPS:
        (
            sum_reward,
            avg_actor_loss,
            avg_critic_loss,
            avg_log_probs,
            avg_alpha,
            steps,
            _,
        ) = run_episode(
            model=model,
            env=env,
            episode_type=(
                EpisodeType.VALIDATION
                if ((episode + 1) % VALIDATION_INTERVAL == 0)
                else EpisodeType.TRAINING
            ),
            model_path=model_path,
        )

        episode += 1
        num_steps += steps

        print(
            f"Episode: {episode} || Reward: {sum_reward} || Actor Loss: {avg_actor_loss} || Critic Loss: {avg_critic_loss} || Log Prob: {avg_log_probs} || Alpha: {avg_alpha} || Total Steps: {num_steps}"
        )

    print(f"Final Time: {time.perf_counter() - start_time}")

    model.save()


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
        target_entropy=(-env.num_actions()),
        discount_factor=SAC_DROQ_DEFAULT_CONFIG.discount_factor,
        tau=SAC_DROQ_DEFAULT_CONFIG.tau,
    )

    if TRAINING:
        run_training(model=model, env=env, model_path=model_path)
    else:
        run_test(model=model, env=env, model_path=model_path)
