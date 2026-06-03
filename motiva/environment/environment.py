from physicsenv.physicsenv import PhysicsEnv
from music.song import Song
import time
import numpy as np

class Environment:
    def __init__(self, song: Song):
        self.physicsenv = PhysicsEnv()
        self.song = song

    def reset(self):
        self.physicsenv.reset()
        self.start_time = time.perf_counter_ns()

        env_obs = self.physicsenv.get_obs()
        song_obs = self.song.sample_at(0)[0]

        return self.get_obs(env_obs, song_obs)

    def step(self, action: np.ndarray):
        env_obs = self.physicsenv.step(action)
        song_obs, done = self.song.sample_at((time.perf_counter_ns() - self.start_time) / 1e9)
        return self.get_obs(env_obs, song_obs), self.get_reward(env_obs[0], song_obs[0]), done
    
    def get_obs(self, env_obs: tuple, song_obs: np.ndarray):
        return np.concatenate((*env_obs, song_obs))
    
    def get_reward(self, piano_actual_state, piano_goal_state):
        # key press reward
        piano_state_error = piano_goal_state - piano_actual_state
        accurate_key_presses = 0.5 * (1 - np.linalg.norm(piano_state_error))
        false_positive_penalty = 0.5 * (1 - np.any(piano_state_error < 0))
        key_press_reward = accurate_key_presses - false_positive_penalty

        # energy penalty

        # finger close to key reward

        return key_press_reward + 0 + 0

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