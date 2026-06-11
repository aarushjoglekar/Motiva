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
NUM_EPISODES = 3
VALIDATION_INTERVAL = 2

# TESTING SETTINGS
SAVE_TO_MIDI = False

# SONG SETTINGS
SONG = Song.from_txt(Song.SOMEWHERE_OVER_THE_RAINBOW)

def run_episode(env: Environment, episode_type: EpisodeType, model_path: str, on_step_end=None):
    observation = env.reset(
        play_audio=(episode_type == EpisodeType.TEST),
        record_midi=(episode_type == EpisodeType.VALIDATION or (episode_type == EpisodeType.TEST and SAVE_TO_MIDI)),
        midi_path=model_path
    )

    while True:
        observation, reward, done = env.step(action=np.zeros(48))

        if on_step_end is not None:
            on_step_end()

        if done:
            return False
        
        if episode_type == EpisodeType.TEST and not env.viewer_running():
            return True

def run_test(env: Environment, model_path:str):
    time.sleep(0.5)

    STEP_DURATION = 1 / Song.RESOLUTION

    def pace():
        sleep_time = ((env.start_time + 1e9 * env.step_count * STEP_DURATION) - time.perf_counter_ns()) / 1e9

        if sleep_time > 0:
            time.sleep(sleep_time)

    closed_viewer = run_episode(env, episode_type=EpisodeType.TEST, model_path=model_path, on_step_end=pace)
    if not closed_viewer:
        time.sleep(2)

def run_training(env: Environment, model_path:str):
    os.makedirs(model_path, exist_ok=True)
    for episode in range(NUM_EPISODES):
        run_episode(env, episode_type=(EpisodeType.VALIDATION if ((episode + 1) % VALIDATION_INTERVAL == 0) else EpisodeType.TRAINING), model_path=model_path)

with Environment(
    SONG, 
    should_render=(not TRAINING), 
    get_time=((lambda env: (env.step_count / Song.RESOLUTION)) if TRAINING else (lambda env: (time.perf_counter_ns() - env.start_time) / 1e9))
) as env:
    DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(DIR, f"ml/models/{MODEL_NAME}")

    if TRAINING:
        run_training(env, model_path)
    else:
        run_test(env, model_path)