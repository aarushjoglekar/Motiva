from physicsenv.physicsenv import PhysicsEnv
from music.song import Song
from music.pianoaudio import PianoAudio
import time
import numpy as np
import helpers.helpers as helpers


class Environment:
    def __init__(self, song: Song, should_render: bool, seed: int):
        self.physicsenv = PhysicsEnv(seed=seed)

        self.song = song
        self.should_render = should_render
        self.piano_audio = None

        if should_render:
            self.physicsenv.render()

    def reset(
        self, play_audio: bool, record_midi: bool, save_midi: bool, midi_file: str
    ):
        self.piano_audio = PianoAudio(
            play_audio=play_audio,
            record_midi=record_midi,
            save_midi=save_midi,
            midi_file=midi_file,
        )

        self.physicsenv.reset()
        self.step_count = 0
        self.start_time = time.perf_counter_ns()

        env_obs = self.physicsenv.get_obs()
        song_obs = self.song.sample_at(0)[0]

        return self.get_obs(env_obs, song_obs)

    def save_piano_audio(self):
        if self.piano_audio is not None:
            return self.piano_audio.save_and_close()

    def step(self, action: np.ndarray):
        self.step_count += 1
        episode_time = self.step_count / Song.RESOLUTION

        env_obs = self.physicsenv.step(action)

        song_obs, fingers_to_keys, truncated = self.song.sample_at(episode_time)

        if self.should_render:
            self.physicsenv.render()

        if self.piano_audio is not None:
            self.piano_audio.update(
                env_obs[0],
                self.physicsenv.data.qvel[self.physicsenv.piano_joint_ids],
                episode_time,
            )

        return (
            self.get_obs(env_obs, song_obs),
            self.get_reward(
                env_obs[0],
                song_obs[: Song.NUM_PIANO_NOTES],
                song_obs[Song.NUM_PIANO_NOTES : Song.NUM_FEATURES],
                fingers_to_keys,
            ),
            truncated,
        )

    def get_obs(self, env_obs: tuple, song_obs: np.ndarray):
        return np.concatenate((*env_obs, song_obs))

    def get_reward(
        self,
        piano_actual_state: np.ndarray,
        piano_goal_state: np.ndarray,
        active_fingers: np.ndarray,
        active_keys: np.ndarray,
    ):
        # key press reward
        key_should_be_pressed = np.where(piano_goal_state == 1)[0]
        piano_state_error = (0.5 * (piano_actual_state[key_should_be_pressed] + 1)) - 1
        accurate_key_presses = (
            0
            if len(piano_state_error) == 0
            else 0.5
            * helpers.proximity_reward(
                np.abs(piano_state_error),
                lower=0,
                upper=0.05,
                margin=0.5,
                value_at_margin=0.1,
            ).mean()
        )

        key_is_sounding = piano_actual_state >= PianoAudio.PRESS_THRESHOLD
        no_false_positive_reward = 0.5 * (
            1 - np.any(key_is_sounding & (piano_goal_state == 0))
        )

        key_press_reward = accurate_key_presses + no_false_positive_reward

        # finger close to key reward
        active_finger_site_ids = self.physicsenv.finger_site_ids[
            np.where(active_fingers == 1)
        ]
        fingertip_positions = self.physicsenv.data.site_xpos[active_finger_site_ids]
        finger_dist_reward = 0

        key_site_ids = self.physicsenv.piano_site_ids[
            active_keys[active_keys >= 0]
        ]  # active keys is in order of active fingers
        active_keys_positions = self.physicsenv.data.site_xpos[key_site_ids]

        dist = np.linalg.norm(fingertip_positions - active_keys_positions, axis=-1)
        finger_dist_reward = (
            0
            if len(active_keys_positions) == 0
            else helpers.proximity_reward(
                dist, lower=0, upper=0.01, margin=0.1, value_at_margin=0.1
            ).mean()
        )

        # energy efficiency penalty
        joint_torques = self.physicsenv.data.qfrc_actuator[
            self.physicsenv.hand_joint_ids
        ]
        joint_velocities = self.physicsenv.data.qvel[self.physicsenv.hand_joint_ids]
        energy_penalty = np.dot(np.abs(joint_torques), np.abs(joint_velocities))

        return key_press_reward + finger_dist_reward - 0.005 * energy_penalty

    def num_actions(self):
        return self.physicsenv.model.nu

    def num_observations(self):
        env_obs = self.physicsenv.get_obs()
        song_obs = self.song.sample_at(0)[0]
        return self.get_obs(env_obs, song_obs).shape[0]

    def render(self):
        return self.physicsenv.render()

    def viewer_running(self):
        return self.physicsenv.viewer_running()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.physicsenv.viewer is not None:
            self.physicsenv.viewer.__exit__(exc_type, exc_val, exc_tb)
            self.physicsenv.viewer = None
