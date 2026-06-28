from music.song import Song
from environment.environment import Environment
from ml.sac_droq import SAC_DROQ
from ml.config import SAC_DROQ_DEFAULT_CONFIG
from datetime import datetime
import time
import os
import torch
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

### SETTINGS
# GENERAL SETTINGS
MODEL_NAME = "twinkle_twinkle_little_star_v2"
SEED = 42
DISABLE_CUDA = False

# TRAINING SETTINGS
TRAINING = True
NUM_STEPS = 1000000
VALIDATION_INTERVAL = 10000

# TESTING SETTINGS
SAVE_TO_MIDI = False

# SONG SETTINGS
SONG_CHOICE = Song.TWINKLE_TWINKLE_LITTLE_STAR
SONG = Song.from_txt(name=SONG_CHOICE)
GROUND_TRUTH = Song.from_midi_string(name=SONG_CHOICE)

torch.manual_seed(SEED)
np.random.seed(SEED)


def run_training(
    model: SAC_DROQ,
    env: Environment,
    model_path: str,
    song: Song,
    ground_truth_song: Song,
    device: str,
):
    os.makedirs(model_path, exist_ok=True)
    num_steps = 0
    episode = 0
    next_validation = VALIDATION_INTERVAL
    num_validations = 0
    f1_score_steps = []
    f1_scores = []
    start_time = time.perf_counter()

    while num_steps < NUM_STEPS:
        validation_episode = (num_steps >= next_validation) or (
            num_steps + song.length >= NUM_STEPS
        )
        if validation_episode:
            next_validation += VALIDATION_INTERVAL
            num_validations += 1

        if validation_episode:
            model.eval()
        else:
            model.train()

        validation_midi_file = os.path.join(
            model_path,
            f"valid-{num_validations}-{datetime.now().strftime('%H-%M')}.mid",
        )
        state = env.reset(
            play_audio=False,
            record_midi=validation_episode,
            midi_file=validation_midi_file,
        )
        state = torch.from_numpy(state).float().to(device)

        warmup_episode = False
        sum_reward = 0.0
        sum_actor_loss = 0.0
        sum_critic_loss = 0.0
        sum_log_prob = 0.0
        sum_alpha = 0.0
        steps = 0

        episode_update_count = 0
        episode_start_time = time.perf_counter()
        while True:
            action, _ = model.select_action(
                state=state, deterministic=validation_episode
            )
            next_obs, reward, truncated = env.step(action=action.detach().cpu().numpy())
            next_state = torch.from_numpy(next_obs).float().to(device)

            if not validation_episode:
                updated = model.update(
                    state=state,
                    next_state=next_state,
                    action=action,
                    reward=reward,
                    truncated=truncated,
                )

                if updated is not None:
                    episode_update_count += model.updates_per_step

                    actor_loss, critic_loss, log_probs, alpha = updated
                    sum_actor_loss += actor_loss
                    sum_critic_loss += critic_loss
                    sum_log_prob += log_probs
                    sum_alpha += alpha
                else:
                    warmup_episode = True

            steps += 1
            sum_reward += reward

            state = next_state

            if truncated:
                break

        episode += 1
        num_steps += steps

        episode_time = time.perf_counter() - episode_start_time

        stats = None
        if warmup_episode:
            stats = "Warmup Episode: No Update Statistics"
        elif validation_episode:
            f1 = None
            midi = env.save_piano_audio()
            if midi is not None:
                f1 = Song.from_midi(name="", midi=midi).compare_to(
                    ground_truth=ground_truth_song
                )
                f1_score_steps.append(num_steps)
                f1_scores.append(f1)
            stats = f"Validation Episode - F1 Score: {f1}"
        else:
            stats = f"Actor Loss: {sum_actor_loss / steps} || Critic Loss: {sum_critic_loss / steps} || Log Prob: {sum_log_prob / steps} || Alpha: {sum_alpha / steps} || Time/Update: {(round(1000 * episode_time / episode_update_count, 2))}ms"

        print(
            f"Episode: {episode} || Reward: {sum_reward} || {stats} || Total Steps: {num_steps}"
        )

    print(f"Train Time: {time.perf_counter() - start_time}")

    plt.plot(f1_score_steps, f1_scores)
    plt.xlabel("Steps")
    plt.ylabel("F1 Score")
    plt.title("F1 Score over Training")
    plt.savefig(
        os.path.join(model_path, f"f1-history-{datetime.now().strftime('%H-%M')}.png")
    )

    model.save()


def run_test(model: SAC_DROQ, env: Environment, model_path: str, device: str):
    time.sleep(0.5)

    def pace():
        sleep_time = (
            (env.start_time + 1e9 * env.step_count / Song.RESOLUTION)
            - time.perf_counter_ns()
        ) / 1e9
        if sleep_time > 0:
            time.sleep(sleep_time)

    model.eval()

    state = env.reset(
        play_audio=True,
        record_midi=SAVE_TO_MIDI,
        midi_file=os.path.join(
            model_path, f"test-{datetime.now().strftime('%H-%M')}.mid"
        ),
    )
    state = torch.from_numpy(state).float().to(device)

    total_reward = 0.0
    closed_viewer = False
    while True:
        with torch.no_grad():
            action, _ = model.select_action(state=state, deterministic=True)
        next_state, reward, truncated = env.step(action=action.detach().cpu().numpy())
        state = torch.from_numpy(next_state).float().to(device)

        total_reward += reward
        pace()

        if truncated:
            break
        if not env.viewer_running():
            closed_viewer = True
            break

    if not closed_viewer:
        time.sleep(2)

    print(f"Test Episode || Total Reward: {total_reward}")


with Environment(SONG, should_render=(not TRAINING)) as env:
    DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(DIR, f"ml/models/{MODEL_NAME}")

    device = "cuda" if torch.cuda.is_available() and not DISABLE_CUDA else "cpu"

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
        target_entropy=(-0.5 * env.num_actions()),
        discount_factor=SAC_DROQ_DEFAULT_CONFIG.discount_factor,
        tau=SAC_DROQ_DEFAULT_CONFIG.tau,
        device=device,
    ).to(device=device)

    if TRAINING:
        run_training(
            model=model,
            env=env,
            model_path=model_path,
            song=SONG,
            ground_truth_song=GROUND_TRUTH,
            device=device,
        )
    else:
        run_test(model=model, env=env, model_path=model_path, device=device)
