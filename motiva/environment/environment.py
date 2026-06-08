from physicsenv.physicsenv import PhysicsEnv
from music.song import Song
from music.pianoaudio import PianoAudio
import time
import numpy as np
import helpers.helpers as helpers

class Environment:
    def __init__(self, song: Song, should_render: bool):
        self.physicsenv = PhysicsEnv()
        self.song = song
        self.should_render = should_render
        self.piano_audio = None

        if should_render:
            self.physicsenv.render()

    def reset(self, play_audio:bool, record_midi:bool, midi_path: str):
        if self.piano_audio is not None:
            self.piano_audio.save_and_close()
        self.piano_audio = PianoAudio(play_audio, record_midi, midi_path)

        self.physicsenv.reset()
        self.start_time = time.perf_counter_ns()

        env_obs = self.physicsenv.get_obs()
        song_obs = self.song.sample_at(0)[0]

        return self.get_obs(env_obs, song_obs)

    def step(self, action: np.ndarray):
        episode_time = (time.perf_counter_ns() - self.start_time) / 1e9

        env_obs = self.physicsenv.step(action)
        song_obs, fingers_to_keys, done = self.song.sample_at(episode_time)

        if self.should_render:
            self.physicsenv.render()

        if self.piano_audio is not None:
            self.piano_audio.update(env_obs[0], self.physicsenv.data.qvel[self.physicsenv.piano_joint_ids], episode_time)

        return self.get_obs(env_obs, song_obs), self.get_reward(env_obs[0], song_obs[:Song.NUM_PIANO_NOTES], song_obs[Song.NUM_PIANO_NOTES:Song.NUM_FEATURES], fingers_to_keys), done
    
    def get_obs(self, env_obs: tuple, song_obs: np.ndarray):
        return np.concatenate((*env_obs, song_obs))
    
    def get_reward(self, piano_actual_state:np.ndarray, piano_goal_state:np.ndarray, active_fingers:np.ndarray, active_keys:np.ndarray):
        # key press reward
        piano_state_error = piano_goal_state - piano_actual_state
        accurate_key_presses = 0.5 * helpers.proximity_reward(
            np.linalg.norm(piano_state_error).item(),
            lower=0,
            upper=0.05,
            margin=0.5,
            value_at_margin=0.1
        )

        max_joint_range = self.physicsenv.model.jnt_range[self.physicsenv.piano_joint_ids, 1]
        key_is_sounding = piano_actual_state >= (max_joint_range - np.deg2rad(0.5))
        false_positive_penalty = 0.5 * np.any(key_is_sounding & (piano_goal_state == 0))

        key_press_reward = accurate_key_presses - false_positive_penalty

        # finger close to key reward
        active_finger_site_ids = self.physicsenv.finger_site_ids[np.where(active_fingers == 1)]
        fingertip_positions = self.physicsenv.data.site_xpos[active_finger_site_ids]
        finger_dist_reward = 0

        key_site_ids = self.physicsenv.piano_site_ids[active_keys[active_keys >= 0]] # active keys is in order of active fingers
        active_keys_positions = self.physicsenv.data.site_xpos[key_site_ids]

        dist = np.linalg.norm(fingertip_positions - active_keys_positions, axis=-1)
        finger_dist_reward = 0 if len(active_keys_positions) == 0 else helpers.proximity_reward(
            dist,
            lower=0,
            upper=0.01,
            margin=0.5,
            value_at_margin=0.1
        ).mean()

        # energy efficiency penalty
        joint_torques = self.physicsenv.data.qfrc_actuator[self.physicsenv.hand_joint_ids]
        joint_velocities = self.physicsenv.data.qvel[self.physicsenv.hand_joint_ids]
        energy_penalty = np.dot(np.abs(joint_torques), np.abs(joint_velocities))

        return key_press_reward + finger_dist_reward - 0.005 * energy_penalty

    def render(self):
        return self.physicsenv.render()

    def viewer_running(self):
        return self.physicsenv.viewer_running()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.physicsenv.viewer is not None:
            self.physicsenv.viewer.close()
            self.physicsenv.viewer = None

        if self.piano_audio is not None and exc_type is None:
            self.piano_audio.save_and_close()