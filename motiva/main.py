from music.song import Song
from environment.environment import Environment
import numpy as np

song = Song.from_txt(Song.CHOPIN_WALTZ_OP69_NO1)

with Environment(song) as env:
    env.reset()
    done = False
    while env.viewer_running() and not done:
        observation, reward, time, done = env.step(action=np.zeros(46)) # add one more for sustain pedal
        env.render()