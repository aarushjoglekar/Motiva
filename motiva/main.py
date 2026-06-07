from music.song import Song
from environment.environment import Environment
import numpy as np
import time

### SETTINGS
# TRAINING SETTINGS
TRAINING = False
NUM_EPISODES = 2
VALIDATION_INTERVAL = 1

# TESTING SETTINGS
SAVE_TO_MIDI = False

# SONG SETTINGS
SONG = Song.from_txt(Song.CHOPIN_WALTZ_OP69_NO1)

TRAINING = 0
VALID = 1
TEST = 2
def run_episode(env: Environment, episode_type: int):
    observation = env.reset()

    while episode_type != TEST or env.viewer_running():
        observation, reward, done = env.step(action=np.zeros(46))
        if done:
            return

def run_test(env: Environment):
    time.sleep(1)
    while env.viewer_running():
        run_episode(env, episode_type=TEST)
        time.sleep(2)

def run_training(env: Environment):
    for episode in range(NUM_EPISODES):
        run_episode(env, episode_type=(VALID if ((episode + 1) % VALIDATION_INTERVAL == 0) else TEST))

with Environment(SONG, should_render=(not TRAINING)) as env:
    if TRAINING:
        run_training(env)
    else:
        run_test(env)