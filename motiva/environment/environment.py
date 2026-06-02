from physicsenv.physicsenv import PhysicsEnv
from music.song import Song
import time
import numpy as np

class Environment:
    def __init__(self, song: Song):
        self.physicsenv = PhysicsEnv()
        self.song = song

    def reset(self):
        self.start_time = time.perf_counter_ns()
        return self.song.sample_at(0)

    def step(self, action: np.ndarray):
        obs = self.physicsenv.step(action)
        song_obs, done = self.song.sample_at((time.perf_counter_ns() - self.start_time) / 1e9)
        return np.concatenate((obs + song_obs)), self.get_reward(), done
    
    def get_reward(self):
        return None

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