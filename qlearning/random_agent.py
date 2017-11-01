from __future__ import unicode_literals

import gym

env = gym.make('Breakout-v4')
env.reset()

while True:
    _,_,finished,_ =env.step(env.action_space.sample())
    env.render()
    if finished:
        env.reset()