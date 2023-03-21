import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np
import random

from collections import namedtuple
from net import PGNet, weight_init

Gamma = 0.99
Epsilon = 0.01
LearningRate = 1e-2
BatchSize = 128
MaxBufferSize = 128 * 128

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReinforceAgent(object):
    def __init__(self, env_dim, action_dim):
        self.env_dim = env_dim
        self.action_dim = action_dim
        self.do_train = True
        self.pg_net = PGNet(env_dim, action_dim)
        weight_init(self.pg_net)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.policy_network.parameters(), lr=LearningRate, momentum=0.9, weight_decay=1e-5)

    def enable_train(self):
        self.do_train = True
        self.cur_reward = 0.0
        self.steps = 0
        self.trajectory = []

    def disable_train(self):
        self.enable_train = False
    
    def reset_env(self):
        self.cur_reward = 0.0
        self.steps = 0

    @torch.no_grad()
    def select_action(self, state):
        if len(self.trajectory) < BatchSize * 2:
            return random.choice(range(self.action_dim))
        state = torch.from_numpy(state)
        # select according to probability distribution
        prob = self.pg_net(state)
        return torch.multinomial(prob, 1).item()

    def add_to_memory(self, state, action, reward, next_state, done):
        if len(self.trajectory) >= MaxBufferSize:
            self.trajectory.pop()
        reward = float(reward)
        self.trajectory.insert(0, Transition(torch.from_numpy(state), 
                                             torch.tensor([action]), 
                                             torch.from_numpy(next_state), 
                                             torch.tensor([reward]), done))

    def copy_new_params(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
    def learn_by_replay(self):
        if len(self.trajectory) <= BatchSize or not self.enable_train:
            return
        # Random choose BATCH_SIZE transitions.
        samples = random.sample(self.trajectory, BatchSize)
        states_batch = torch.stack([t.state for t in samples])
        next_states_batch = torch.stack([t.next_state for t in samples])
        actions_batch = torch.cat([t.action for t in samples]).to(torch.int64)
        rewards_batch = torch.stack([t.reward for t in samples])
        done_batch = torch.cat([torch.Tensor([t.done]) for t in samples])

        # Update pg net, maximize Log-Likelyhood, minimize cross entropy

