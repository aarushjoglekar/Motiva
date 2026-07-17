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
import logging

matplotlib.use("agg")
import matplotlib.pyplot as plt

### SETTINGS
# GENERAL SETTINGS
MODEL_NAME = "another_love"
SEED = 42
DISABLE_CUDA = False

# TRAINING SETTINGS
TRAINING = True
NUM_STEPS = 5000000
VALIDATION_INTERVAL = 10000
SAVE_TO_MIDI_VALID = False

# TESTING SETTINGS
SAVE_TO_MIDI_TEST = False

# SONG SETTINGS
SONG_CHOICE = Song.ANOTHER_LOVE
SONG = Song.from_txt(name=SONG_CHOICE)

torch.manual_seed(SEED)
np.random.seed(SEED)

torch.set_float32_matmul_precision("high")
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)


def run_training(
    model: SAC_DROQ,
    env: Environment,
    model_path: str,
    device: str,
):
    os.makedirs(model_path, exist_ok=True)
    num_steps = 0
    train_episode = 0
    f1_score_steps = []
    f1_scores = []

    total_train_time = 0
    total_loop_time = 0

    train_data_path = os.path.join(model_path, "train_data.txt")
    if os.path.exists(train_data_path):
        with open(train_data_path) as f:
            header = f.readline()

        train_episode = int(header.split("train_episodes=")[1].split(" ")[0])
        num_steps = int(header.split("num_steps=")[1].split(" ")[0])
        total_train_time = float(header.split("total_train_time=")[1].split(" ")[0])
        total_loop_time = float(header.split("total_loop_time=")[1].strip())

        data = np.loadtxt(train_data_path)
        f1_score_steps = data[:, 0].tolist()
        f1_scores = data[:, 1].tolist()
        print(f"Resumed with {len(f1_score_steps)} existing F1 scores")

    next_validation = num_steps + VALIDATION_INTERVAL
    total_target_steps = num_steps + NUM_STEPS

    start_time = time.perf_counter()
    last_start_train_time = time.perf_counter()
    train_checkpoint_time = 0

    while True:
        validation_episode = num_steps >= next_validation

        if validation_episode:
            train_checkpoint_time += time.perf_counter() - last_start_train_time

            next_validation += VALIDATION_INTERVAL
            stats = run_validation_episode(
                model=model,
                env=env,
                model_path=model_path,
                f1_score_steps=f1_score_steps,
                f1_scores=f1_scores,
                num_steps=num_steps,
                device=device,
            )

            last_start_train_time = time.perf_counter()
        else:
            train_episode += 1
            stats, num_steps = run_train_episode(
                model=model,
                env=env,
                max_steps=(total_target_steps - num_steps),
                num_steps=num_steps,
                episode=train_episode,
                device=device,
            )

        print(stats)

        if num_steps >= total_target_steps:
            train_checkpoint_time += time.perf_counter() - last_start_train_time
            break

    stats = run_validation_episode(
        model=model,
        env=env,
        model_path=model_path,
        f1_score_steps=f1_score_steps,
        f1_scores=f1_scores,
        num_steps=num_steps,
        device=device,
    )
    print(stats)

    loop_checkpoint_time = time.perf_counter() - start_time
    total_train_time += train_checkpoint_time
    total_loop_time += loop_checkpoint_time
    print(f"Time Statistics:\n\
            \tTrain Checkpoint Time: {train_checkpoint_time / 3600} hrs\n\
            \tLoop Checkpoint Time: {loop_checkpoint_time / 3600} hrs\n\
            \tTotal Train Time: {total_train_time / 3600} hrs\n\
            \tTotal Loop Time: {total_loop_time / 3600} hrs")

    np.savetxt(
        train_data_path,
        np.column_stack([f1_score_steps, f1_scores]),
        header=f"train_episodes={train_episode} || num_steps={num_steps} || total_train_time={total_train_time} || total_loop_time={total_loop_time}",
        fmt="%.6f",
    )

    plt.plot(f1_score_steps, f1_scores)
    plt.xlabel("Steps")
    plt.ylabel("F1 Score")
    plt.title("F1 Score over Training")
    plt.savefig(os.path.join(model_path, "f1-history.png"))

    model.save()


def run_train_episode(model: SAC_DROQ, env: Environment, max_steps: int, num_steps: int, episode: int, device: str):
    model.train()

    state = env.reset(
        play_audio=False,
        record_midi=False,
        save_midi=False,
        midi_file="",
    )
    state = torch.from_numpy(state).float().to(device)

    warmup_episode = False
    sum_reward = 0
    sum_actor_loss = torch.zeros((), device=device)
    sum_critic_loss = torch.zeros((), device=device)
    sum_log_prob = torch.zeros((), device=device)
    sum_alpha = torch.zeros((), device=device)
    steps = 0

    episode_update_count = 0
    episode_start_time = time.perf_counter()
    while True:
        action, _ = model.select_action(state=state, deterministic=False)
        next_obs, reward, truncated = env.step(action=action.detach().cpu().numpy())
        next_state = torch.from_numpy(next_obs).float().to(device)

        updated = model.update(
            state=state, next_state=next_state, action=action, reward=reward
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

        if truncated or steps >= max_steps:
            break

    episode_time = time.perf_counter() - episode_start_time

    stats = f"Episode: {episode} || Reward: {sum_reward} || "
    if warmup_episode:
        stats += "Warmup Episode: No Update Statistics"
    else:
        stats += f"Actor Loss: {(sum_actor_loss / steps).item()} || Critic Loss: {(sum_critic_loss / steps).item()} || Log Prob: {(sum_log_prob / steps).item()} || Alpha: {(sum_alpha / steps).item()} || Time/Update: {(round(1000 * episode_time / episode_update_count, 2))}ms"
    num_steps += steps
    stats += f" || Total Steps: {num_steps}"

    return stats, num_steps


def run_validation_episode(
    model: SAC_DROQ,
    env: Environment,
    model_path: str,
    f1_score_steps: list[int],
    f1_scores: list[float],
    num_steps: int,
    device: str,
):
    model.eval()

    validation_midi_file = os.path.join(
        model_path,
        f"valid-{num_steps}.mid",
    )
    state = env.reset(
        play_audio=False,
        record_midi=True,
        save_midi=SAVE_TO_MIDI_VALID,
        midi_file=validation_midi_file,
    )
    state = torch.from_numpy(state).float().to(device)

    sum_reward = 0

    while True:
        action, _ = model.select_action(state=state, deterministic=True)
        next_obs, reward, truncated = env.step(action=action.detach().cpu().numpy())
        next_state = torch.from_numpy(next_obs).float().to(device)
        sum_reward += reward

        state = next_state

        if truncated:
            break

    f1 = None
    precision = None
    recall = None
    midi = env.save_piano_audio()
    if midi is not None:
        precision, recall, f1 = Song.from_midi(name="", midi=midi).compare_to(
            ground_truth=SONG
        )

        f1_score_steps.append(num_steps)
        f1_scores.append(f1)

    stats = f"Validation Episode - Reward: {sum_reward} F1: {f1}, Precision: {precision}, Recall: {recall}"

    return stats


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
        record_midi=True,
        save_midi=SAVE_TO_MIDI_TEST,
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

    additional = ""
    midi = env.save_piano_audio()
    if midi is not None:
        precision, recall, f1 = Song.from_midi(name="", midi=midi).compare_to(
            ground_truth=SONG
        )
        additional = f" || Precision: {precision} || Recall: {recall} || F1: {f1}"

    print(f"Test Episode || Total Reward: {total_reward}{additional}")


with Environment(SONG, should_render=(not TRAINING), seed=SEED) as env:
    DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(DIR, f"ml/models/{MODEL_NAME}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and not DISABLE_CUDA else "cpu"
    print(f"Running on device: {device}")

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
            device=device,
        )
    else:
        run_test(
            model=model,
            env=env,
            model_path=model_path,
            device=device,
        )
