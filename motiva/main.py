from music.song import Song
from environment.environment import Environment
import numpy as np
import time

song = Song.from_txt(Song.CHOPIN_WALTZ_OP69_NO1)

with Environment(song, True) as env:

    if True: # should_render based on training
        time.sleep(1)

    observation = env.reset()
    done = False
    while env.viewer_running() and not done:
        observation, reward, done = env.step(action=np.zeros(46))

        if done:
            time.sleep(2)