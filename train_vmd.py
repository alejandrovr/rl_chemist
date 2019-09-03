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
from net import DQN


#GYM SETUP AND DEVICES
env = gym.make('tictac4-v0')

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

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


#RENDERING IMAGES
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    #screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = env.state
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return screen.unsqueeze(0).unsqueeze(0).float().to(device)


#env.reset()

#GETTING READY
BATCH_SIZE = 10 #128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space

policy_actions = ['rotx','roty','rotz','switch_dir','movedih','nextdih']
n_actions = len(policy_actions)

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
#policy_net.load_state_dict(torch.load('./first_working_agent.torch'))
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

#optimizer = optim.RMSprop(policy_net.parameters(), lr=0.0001)
optimizer = optim.SGD(policy_net.parameters(), lr=0.0001)
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
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            pred_action_reward = policy_net(state)
            #['rotx','roty','rotz','switch_dir','movedih','nextdih']
            print('POLICY',pred_action_reward)
            policy_pred_rew.append(pred_action_reward)
            return pred_action_reward.max(1)[1].view(1, 1)
    else:
        print('RANDOM')
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def plot_action_pred_val():
    plt.figure(1)
    actions_val = []
    for row in policy_pred_rew:
        pred_val_action = row.cpu().numpy().tolist()[0]
        actions_val.append(pred_val_action)

    action_rotx = [i[0] for i in actions_val]
    if len(action_rotx)>0:
        print("HERE!")
        plt.plot(np.array(action_rotx),color="red")
        plt.show()

#the optimizer
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
    #print('action_batch',action_batch)
    #print('reward_batch',reward_batch)
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
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    criterion = nn.MSELoss()
    #print('expected_action_values',expected_state_action_values)
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    #NOTE: state_action_values  and expected_action_values tends to go to 1 in all cells
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


#RUN THE THING
num_episodes = 500
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
            #episode_durations.append(t + 1)
            episode_durations.append(reward)
            plot_durations()
            #plot_action_pred_val()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        
    env.reset()
    
print('Complete')
plt.ioff()
plt.show()




