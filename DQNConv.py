import torch
from torch import nn
from torch.nn import functional as F
from collections import namedtuple

#Transition, the format of the memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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
    def __init__(self, action_dim, n_latent_var, lr, betas, gamma):
        self.lr = lr
        self.betas = betas
        self.gamma = torch.tensor(gamma)

        self.policy_net = ConvNet(action_dim, n_latent_var)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr, betas=betas)
        self.target_net = ConvNet(action_dim, n_latent_var)
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

    def __init__(self, outputSize, n_latent_var):
        super(ConvNet, self).__init__()

        self.strategy = EpsilonGreedyStrategy(0.99, 0.05, 3000)
        self.randPolicy = {"Rand": 0, "Policy": 0}
        self.current_step = 0

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(5, 5), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5, 5), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5, 5), stride=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.hidden = nn.Linear(1600, n_latent_var) #64 (out_channel)*5*5 (remaining )
        self.output = nn.Linear(n_latent_var, outputSize)

    def forward(self, t):
        t = self.conv1(t).float()
        t = F.tanh(self.conv2(t)).float()
        t = F.tanh(self.conv3(t)).float()
        t = F.tanh(self.conv4(t)).float()
        t = F.tanh(self.conv5(t)).float()
        t = torch.flatten(t, start_dim=1)
        t = F.tanh(self.hidden(t)).float()
        return self.output(t.view(t.size(0), -1))

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