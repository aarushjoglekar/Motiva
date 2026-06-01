from environment.environment import Environment
import numpy as np

with Environment() as env:
    while env.viewer_running():
        env.step(action=np.zeros(46))
        env.render()