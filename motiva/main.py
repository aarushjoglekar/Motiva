from environment.environment import Environment

with Environment() as env:
    while env.viewer_running():
        env.step()
        env.render()