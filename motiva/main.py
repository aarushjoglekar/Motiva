from music.song import Song
from environment.environment import Environment
from ml.sac_droq import SAC_DROQ
from ml.config import SAC_DROQ_DEFAULT_CONFIG
from datetime import datetime
import time
import os
import json
import torch
import numpy as np
import matplotlib
import logging

matplotlib.use("agg")
import matplotlib.pyplot as plt

### SETTINGS
# GENERAL SETTINGS
MODEL_NAME = "dynamics/level_1/payphone_wiz_khalifa"
SEED = 42
DISABLE_CUDA = False

# TRAINING SETTINGS
TRAINING = True
NUM_STEPS = 5000000
VALIDATION_INTERVAL = 10000
SAVE_TO_MIDI_VALID = False
USE_DYNAMICS_DATA = True
USE_FINGERING_LABELS = True
TRAIN_SONGS = [Song.from_txt(song=Song.PAYPHONE_WIZ_KHALIFA)]

# TESTING SETTINGS
SAVE_TO_MIDI_TEST = False
TEST_SONG = Song.from_txt(Song.PAYPHONE_WIZ_KHALIFA)

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
    total_train_time = 0
    total_loop_time = 0
    eval_history = dict()

    song_keys = [song.name for song in TRAIN_SONGS]

    train_state_path = os.path.join(model_path, "train_state.json")
    if os.path.exists(train_state_path):
        with open(train_state_path) as f:
            train_state = json.load(f)

        if set(train_state["song_keys"]) != set(song_keys):
            raise ValueError("Song set changed since this model started training!")

        song_keys = train_state["song_keys"]
        train_episode = train_state["train_episodes"]
        num_steps = train_state["num_steps"]
        total_train_time = train_state["total_train_time"]
        total_loop_time = train_state["total_loop_time"]
        eval_history = train_state["eval_history"]

        print(
            f"Resumed with {len(list(eval_history.values())[0]['steps'])} existing validation checkpoints."
        )
    else:
        for song in TRAIN_SONGS:
            eval_history[song.name] = {
                "steps": [],
                "f1": [],
                "precision": [],
                "recall": [],
                "dynamics_score": [],
                "match_rate": [],
            }

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
                eval_history=eval_history,
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
        eval_history=eval_history,
        num_steps=num_steps,
        device=device,
    )
    print(stats)

    loop_checkpoint_time = time.perf_counter() - start_time
    total_train_time += train_checkpoint_time
    total_loop_time += loop_checkpoint_time
    print(
        f"Time Statistics:\n  Train Checkpoint Time: {train_checkpoint_time / 3600} hrs\n  Loop Checkpoint Time: {loop_checkpoint_time / 3600} hrs\n  Total Train Time: {total_train_time / 3600} hrs\n  Total Loop Time: {total_loop_time / 3600} hrs"
    )

    train_state = {
        "song_keys": song_keys,
        "train_episodes": train_episode,
        "num_steps": num_steps,
        "total_train_time": total_train_time,
        "total_loop_time": total_loop_time,
        "eval_history": eval_history,
    }
    with open(train_state_path, "w") as f:
        json.dump(train_state, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for song_key, history in eval_history.items():
        ax1.plot(history["steps"], history["f1"], label=song_key)
        
        valid = [
            (step, dynamics_score) for step, dynamics_score in zip(history["steps"], history["dynamics_score"])
            if dynamics_score is not None
        ]
        if len(valid) != 0:
            dyn_steps, dyn_scores = zip(*valid)
            ax2.plot(dyn_steps, dyn_scores, label=song_key)
        
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("F1 Score")
    ax1.set_title("F1 Score over Training")
    ax1.legend()

    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Dynamics Score")
    ax2.set_title("Dynamics Score over Training")
    ax2.legend()
    
    fig.tight_layout()
    fig.savefig(os.path.join(model_path, "eval_history.png"))
    plt.close(fig)

    model.save()


def run_train_episode(
    model: SAC_DROQ,
    env: Environment,
    max_steps: int,
    num_steps: int,
    episode: int,
    device: str,
):
    model.train()

    state = env.reset(
        song=None,
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

    stats = f"Episode: {episode} || Reward: {round(sum_reward, 2)} || "
    if warmup_episode:
        stats += "Warmup Episode: No Update Statistics"
    else:
        stats += f"Actor Loss: {round((sum_actor_loss / steps).item(), 2)} || Critic Loss: {round((sum_critic_loss / steps).item(), 2)} || Log Prob: {round((sum_log_prob / steps).item(), 2)} || Alpha: {round((sum_alpha / steps).item(), 2)} || Time/Update: {(round(1000 * episode_time / episode_update_count, 2))}ms"
    num_steps += steps
    stats += f" || Total Steps: {num_steps} || Song: {env.current_song.name}"

    return stats, num_steps


def run_validation_episode(
    model: SAC_DROQ,
    env: Environment,
    model_path: str,
    eval_history: dict,
    num_steps: int,
    device: str,
):
    model.eval()
    stats = "Validation Episode:"

    for song in TRAIN_SONGS:
        validation_midi_file = os.path.join(
            model_path,
            f"valid_{song.name}_{num_steps}.mid",
        )
        state = env.reset(
            song=song,
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
            precision, recall, f1, dynamics_score, match_rate = Song.from_midi(
                name="", type="", should_add_start_buffer=False, midi=midi
            ).compare_to(ground_truth=song)

            eval_history[song.name]["steps"].append(num_steps)
            eval_history[song.name]["f1"].append(f1)
            eval_history[song.name]["precision"].append(precision)
            eval_history[song.name]["recall"].append(recall)
            eval_history[song.name]["dynamics_score"].append(dynamics_score)
            eval_history[song.name]["match_rate"].append(match_rate)

        stats += f"\n  Song: {song.name}\n    Reward: {round(sum_reward, 2)}\n    F1: {round(f1, 2) if f1 is not None else None}, Precision: {round(precision, 2) if precision is not None else None}, Recall: {round(recall, 2) if recall is not None else None}\n    Dynamics Score: {dynamics_score}, Match Rate: {match_rate}"

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
        song=TEST_SONG,
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
        precision, recall, f1, dynamics_score, match_rate = Song.from_midi(
            name="", type="", should_add_start_buffer=False, midi=midi
        ).compare_to(ground_truth=TEST_SONG)
        additional = f" || Precision: {precision} || Recall: {recall} || F1: {f1} || Dynamics Score: {dynamics_score} || Match Rate: {match_rate}"

    print(f"Test Episode || Total Reward: {total_reward}{additional}")


with Environment(
    songs=TRAIN_SONGS,
    use_fingering_labels=USE_FINGERING_LABELS,
    use_dynamics_data=USE_DYNAMICS_DATA,
    should_render=(not TRAINING),
    never_play_audio=TRAINING,
    seed=SEED,
) as env:
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
