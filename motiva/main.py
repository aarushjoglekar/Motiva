from environment.environment import Environment
from music.song import Song
import numpy as np

song = Song.from_txt(Song.CHOPIN_WALTZ_OP69_NO1)

with Environment() as env:
    while env.viewer_running():
        env.step(action=np.zeros(46)) # add one more for sustain pedal
        env.render()