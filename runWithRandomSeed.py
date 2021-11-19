import os
import random
import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from filelock import FileLock
import numpy as np
from collections import namedtuple
from collections import Counter
from collections import deque
import ray
import sys, time
from DQNConv import DQN as CNNDQN

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
        self.optimizer.step()


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
        t = t.float()# t = torch.flatten(t, start_dim=1).float()
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


@ray.remote(max_restarts=-1, max_task_retries=-1, memory=1024*1024*1024)
class DataWorker(object):
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, i, seedList, numWorkers, gymseed):
        #Hyper parameters
        self.gymseed = gymseed
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_latent_var = n_latent_var
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        #Environment
        self.env = gym.make("CarRacing-v0")
        #Otherparameters
        self.id = i
        self.lastRew = 0.0
        self.num = 0
        self.rewards = []
        self.seedList = seedList
        self.numWorkers = numWorkers
        self.device = "cuda:{}".format((i//2)+1)
        print(self.device)

    def rewEnv(self):
        seed = self.seedList[self.num] + self.id
        #Fix the random seeds
        torch.manual_seed(seed)
        self.env.seed(self.gymseed)
        np.random.seed(seed)

        #Model creation
        self.DQN = CNNDQN(self.action_dim, self.n_latent_var, self.lr, self.betas, self.gamma).to(self.device)

        #Perform actions in environment
        rewards = self.performNActions(1000)

        #Append rewards to file
        self.rewards.append(rewards)

        #Increase the number, and if it is x00 it will write rewards
        self.num += self.numWorkers
        return [rewards, seed]

    def performNActions(self, N):
        memory = Memory()
        state = self.env.reset()
        totalRew = 0.0
        for t in range(N):
            prevState = torch.from_numpy(state.copy())
            s = self.numpyToTensor(state)
            action = self.DQN.policy_net(s)
            action = torch.argmax(action, dim=-1)
            actionTranslated = self.translateAction(action)
            state, rew, done, info = self.env.step(actionTranslated)
            totalRew += rew
            # prevState = torch.unsqueeze(prevState, 0)
            # stateMem = torch.unsqueeze(torch.from_numpy(state.copy()), 0)
            rew = torch.unsqueeze(torch.tensor(rew), 0)
            memory.push(prevState, action, state, rew)
            if done:
                print(t, totalRew)
                break

        self.lastRew = totalRew
        return totalRew

    def getRews(self):
        return self.lastRew

    def numpyToTensor(self, state):
        state = self.intTofloat(state)
        s = np.expand_dims(state, axis=0)
        s = np.swapaxes(s, 1, -1)
        return torch.from_numpy(s.copy()).float().to(self.device)

    def intTofloat(self, state):
        return state / 255.0

    def translateAction(self, action):
        actionDict = {0: np.array([0, 1.0, 0]), 1: np.array([-1.0, 0, 0]), 2: np.array([0, 0, 1]),
                      3: np.array([1.0, 0, 0])}
        return actionDict[action.item()]



errorFile = open("error.txt", mode="a")
start = time.process_time()
seedList = [i for i in range(10)]
numActions = 4
stateDim = 96 * 96 * 3
n_latent_var = 256
lr = 0.002
betas = (0.9, 0.999)
gamma = 0.9
num_workers = 30  # Set gpu to num_workers//num_gpu
env = gym.make("CarRacing-v0")
gymSeeds = [0, 1, 2, 3, 4]

ray.init(ignore_reinit_error=True)
for s in gymSeeds:
    workers = [DataWorker.remote(stateDim, numActions, n_latent_var, lr, betas, gamma, x, seedList, num_workers, s)
               for x in range(num_workers)]

    start = seedList[0]
    stop = seedList[-1] + 1
    print(start, stop)
    print(num_workers)

    fileOpen = open("rewards{}.csv".format(s), "w")
    print("Running synchronous parameter server training.")
    for i in range(seedList[0], seedList[-1]+1, num_workers):
        print(s, i)
        try:
            rewards = [worker.rewEnv.remote()
                    for worker in workers]
            x = (ray.get(rewards))
            for r in x:
                print(r)
                fileOpen.write(str(r[0]).format('{:3f}') + "," + str(r[1]).format('{:2f}') + ",,")

            fileOpen.write("\n")
            del rewards
        except Exception as e:
            errorFile.write(e)
            errorFile.write(i)
            sys.exit(1)


        time.sleep(0.5)
        print(time.process_time()-start)

ray.shutdown()