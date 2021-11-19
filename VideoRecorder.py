import os
import cv2
import numpy as np
from PIL import Image
from collections import namedtuple
import random
import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

#Transition, the format of the memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

#Memory used for storing the state, action, nextState, reward
class Memory(object):

    def __init__(self, capacity=7000):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#Used for greedy selection of action for the DQN (starts with 99% random actions, end at 5% for example)
class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        current_rate = self.start ** (current_step // self.decay)
        if current_rate < self.end:
            return self.end
        else:
            return current_rate

#AI model
class DQN():
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma):
        self.lr = lr
        self.betas = betas
        self.gamma = torch.tensor(gamma)

        self.policy_net = ConvNet(state_dim, action_dim, n_latent_var)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr, betas=betas)
        self.target_net = ConvNet(state_dim, action_dim, n_latent_var)
        self.policy_net = self.policy_net.float()
        self.target_net = self.target_net.float()

        self.MseLoss = nn.MSELoss()

    def update(self, memory, BATCH_SIZE):
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch)
        state_action_values = state_action_values.gather(dim=-1, index=action_batch.view(64, 1))

        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)


class ConvNet(nn.Module):
    """Small ConvNet for MNIST."""

    def __init__(self, stateDim, outputSize, n_latent_var):
        super().__init__()
        self.strategy = EpsilonGreedyStrategy(0.99, 0.05, 3000)
        self.randPolicy = {"Rand": 0, "Policy": 0}
        self.current_step = 0
        self.num_actions = outputSize
        self.fc1 = nn.Linear(in_features=stateDim, out_features=n_latent_var).float()
        self.fc2 = nn.Linear(in_features=n_latent_var, out_features=n_latent_var).float()
        self.out = nn.Linear(in_features=n_latent_var, out_features=outputSize).float()

    def forward(self, t):
        t = torch.flatten(t, start_dim=1).float()
        t = self.fc1(t).float()
        t = F.relu(t).float()
        t = self.fc2(t).float()
        t = F.relu(t).float()
        t = self.out(t).float()
        return t

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)

def runner():
    numActions = 4
    stateDim = 96 * 96 * 3
    n_latent_var = 64
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.9
    env = gym.make("CarRacing-v0")
    torch.manual_seed(694917)
    model = DQN(stateDim, numActions, n_latent_var, lr, betas, gamma)
    performAnEpoch(env, model)
    env.close()

def performAnAction(env, model, state, num):
    # state = torch.from_numpy(state.copy())
    s = numpyToTensor(state)
    im = Image.fromarray(state)
    im.save("images/filename{}.jpeg".format(num))
    action = model.policy_net(s)
    action = torch.argmax(action, dim=-1)
    actionTranslated = translateAction(action)
    state_next, reward, done, info = env.step(actionTranslated)
    env.render()
    return state_next

def performAnEpoch(env, model):
    state = env.reset()
    done = False
    num=0
    while not done and num <=1000:
        state = performAnAction(env, model, state, num)
        num += 1

def numpyToTensor(state):
    s = np.expand_dims(state, axis=0)
    s = np.swapaxes(s, 1, -1)
    return torch.from_numpy(s.copy())

def translateAction(action):
    actionDict = {0: np.array([0, 1.0, 0]), 1: np.array([-1.0, 0, 0]), 2: np.array([0, 0, 1]),
                  3: np.array([1.0, 0, 0])}
    return actionDict[action.item()]

def listFiles():
    files = os.listdir("images/")
    output = []

    for a in range(len(files)):
        output.append("images/filename{}.jpeg".format(a))
    return output

def createImageList(files):
    output = []
    for f in files:
        print(f, os.path.exists(f))
        img = cv2.imread(f)
        output.append(img)

    print(len(output), output)
    height, width, channel = output[0].shape
    size = (width, height)
    return output, size

def createVideo(imgList, size):
    out = cv2.VideoWriter('video/bestResult.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for img in imgList:
        out.write(img)
    out.release()

files = listFiles()
imageList, size = createImageList(files)
createVideo(imageList, size)
