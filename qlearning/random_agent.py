from __future__ import unicode_literals

import gym

env = gym.make('Breakout-v4')
env.reset()
while True:
    print "lives before", env.unwrapped.ale.lives()
    _,_,finished,_ =env.step(env.action_space.sample())
    env.render()
    print "lives after", env.unwrapped.ale.lives()
    if finished:
        env.reset()