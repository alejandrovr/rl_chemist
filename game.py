#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 21:53:35 2019

@author: alejandro
"""
from htmd.ui import *
import random
import gym
import gym_tictac4


actions = ['rotx','roty','rotz','scalein','scaleout','submit']    
env = gym.make('tictac4-v0')

n_steps = 50
for i in range(n_steps):
    next_move = random.choice(actions)
    abc = env.step(next_move)
    if env.done == 1:
        break

    
print('Obtained reward:',abc[1])



