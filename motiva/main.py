from music.song import Song
from environment.environment import Environment
import numpy as np
import time

song = Song.from_txt(Song.CHOPIN_WALTZ_OP69_NO1)

with Environment(song) as env:
    env.render()
    time.sleep(1)

    env.reset()
    done = False
    while env.viewer_running() and not done:
        observation, reward, done = env.step(action=np.zeros(46))
        env.render()

        if done:
            time.sleep(2)