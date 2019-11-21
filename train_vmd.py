import gym
import gym_tictac4
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from net import DQN, XRayMan
from torchvision import datasets, models, transforms

#GYM SETUP AND DEVICES
env = gym.make('tictac4-v0')
import os
from glob import glob
[os.remove(i) for i in glob("/home/alejandro/rl_chemist/debug_images/*.png")]

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#REPLAY MEMORY SECTION
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    #screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = env.state
    screen = torch.from_numpy(screen)
    #import matplotlib.pyplot as plt
    #plt.imshow(screen)
    #plt.show()
    #input('next?')
    return screen.unsqueeze(0).transpose(1,3).float().to(device)


env.reset()

#GETTING READY
BATCH_SIZE = 32 #128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 20

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
policy_actions = env.available_actions
n_actions = len(policy_actions)

policy_net = XRayMan()
policy_net.wrap_up[2] = nn.Linear(512, n_actions)
policy_net.to(device)


target_net = XRayMan()
target_net.wrap_up[2] = nn.Linear(512, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.to(device)
target_net.eval()

optimizer = optim.SGD(policy_net.parameters(), lr=0.01)
memory = ReplayMemory(1000) #cartopole 10000

steps_done = 0
policy_pred_rew = []

def select_action(state):
    global steps_done
    global policy_pred_rew
    sample = random.random()
    #The more actions that you take, the better the net is at predicting the expected reward
    #so as you progress, try to use more frequentely the policy net suggested action (exploit vs explore)
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            pred_action_reward = policy_net(state)
            prednp = pred_action_reward.cpu().detach().numpy().flatten().tolist()
            best_action_pred = env.available_actions[prednp.index(max(prednp))]

            #print([i for i in zip(prednp,env.available_actions)])
            policy_pred_rew.append(pred_action_reward)
            return pred_action_reward.max(1)[1].view(1, 1)
    else:
        acran = random.randrange(n_actions)
        #print('RANDOM',acran)
        return torch.tensor([[acran]], device=device, dtype=torch.long)


episode_durations = []


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = next_state_values + reward_batch #(next_state_values * GAMMA) + reward_batch
    #print('\n\nAC:',action_batch.cpu().detach().numpy().flatten().tolist())
    #print('GT:',expected_state_action_values.cpu().detach().numpy().flatten().tolist())
    #print('PD:',state_action_values.cpu().detach().numpy().flatten().tolist())
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    loss.backward()
    print('Loss:',loss.item())
    optimizer.step()

    optimizer.zero_grad()

#RUN THE THING
num_episodes = 100
for i_episode in range(num_episodes):
    # Initialize the environment and state
    current_screen = get_screen() #this is zero, right?
    state = current_screen
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(reward)
            #plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        
    env.reset()
    
print('Complete')

plt.plot(episode_durations)
plt.show()
torch.save(target_net.state_dict(), './new_rl_agent.torch')




