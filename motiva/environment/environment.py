from physicsenv.physicsenv import PhysicsEnv
from music.song import Song
import time
import numpy as np
import helpers.helpers as helpers

class Environment:
    def __init__(self, song: Song, should_render: bool):
        self.physicsenv = PhysicsEnv()
        self.song = song
        self.should_render = should_render

        if should_render:
            self.physicsenv.render()

    def reset(self):
        self.physicsenv.reset()
        self.start_time = time.perf_counter_ns()

        env_obs = self.physicsenv.get_obs()
        song_obs = self.song.sample_at(0)[0]

        return self.get_obs(env_obs, song_obs)

    def step(self, action: np.ndarray):
        env_obs = self.physicsenv.step(action)
        song_obs, fingers_to_keys, done = self.song.sample_at((time.perf_counter_ns() - self.start_time) / 1e9)

        if self.should_render:
            self.physicsenv.render()

        return self.get_obs(env_obs, song_obs), self.get_reward(env_obs[0], song_obs[:Song.NUM_PIANO_NOTES], song_obs[Song.NUM_PIANO_NOTES:Song.NUM_FEATURES], fingers_to_keys), done # type: ignore
    
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
        false_positive_penalty = 0.5 * np.any(piano_state_error < 0)
        key_press_reward = accurate_key_presses - false_positive_penalty

        # finger close to key reward
        active_finger_site_ids = self.physicsenv.finger_site_ids[np.where(active_fingers == 1)]
        fingertip_positions = self.physicsenv.data.site_xpos[active_finger_site_ids]
        finger_dist_reward = 0

        key_site_ids = self.physicsenv.piano_site_ids[active_keys[active_keys >= 0]] # active keys is in order of active fingers
        active_keys_positions = self.physicsenv.data.site_xpos[key_site_ids]

        dist = np.linalg.norm(fingertip_positions - active_keys_positions, axis=-1)
        finger_dist_reward = 0 if len(active_fingers) == 0 else helpers.proximity_reward(
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

    def __exit__(self, *args):
        if self.physicsenv.viewer is not None:
            self.physicsenv.viewer.close()
            self.physicsenv.viewer = None