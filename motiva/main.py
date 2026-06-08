from music.song import Song
from environment.environment import Environment
from environment.episodetype import EpisodeType
import numpy as np
import time
import os

### SETTINGS
# GENERAL SETTINGS
MODEL_NAME = "motiva"

# TRAINING SETTINGS
TRAINING = False
NUM_EPISODES = 2
VALIDATION_INTERVAL = 1

# TESTING SETTINGS
SAVE_TO_MIDI = False

# SONG SETTINGS
SONG = Song.from_txt(Song.CHOPIN_WALTZ_OP69_NO1)

def run_episode(env: Environment, episode_type: EpisodeType, model_path: str):
    observation = env.reset(
        play_audio=(episode_type == EpisodeType.TEST),
        record_midi=(episode_type == EpisodeType.VALIDATION or (episode_type == EpisodeType.TEST and SAVE_TO_MIDI)),
        midi_path=model_path
    )

    while episode_type != EpisodeType.TEST or env.viewer_running():
        observation, reward, done = env.step(action=np.zeros(46))
        if done:
            return

def run_test(env: Environment, model_path:str):
    time.sleep(1)
    run_episode(env, episode_type=EpisodeType.TEST, model_path=model_path)
    time.sleep(2)

def run_training(env: Environment, model_path:str):
    os.makedirs(model_path, exist_ok=True)
    for episode in range(NUM_EPISODES):
        run_episode(env, episode_type=(EpisodeType.VALIDATION if ((episode + 1) % VALIDATION_INTERVAL == 0) else EpisodeType.TEST), model_path=model_path)

with Environment(SONG, should_render=(not TRAINING)) as env:
    DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(DIR, f"ml/models/{MODEL_NAME}")

    if TRAINING:
        run_training(env, model_path)
    else:
        run_test(env, model_path)