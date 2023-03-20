import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from collections import namedtuple
import random

Gamma = 0.99
Epsilon = 0.01
LearningRate = 1e-3
BatchSize = 128
MaxBufferSize = 4096

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class DQNPolicy(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DQNPolicy, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x


class DQNAgent(object):
    def __init__(self, env_dim, action_dim):
        self.env_dim = env_dim
        self.action_dim = action_dim
        self.do_train = True
        self.target_network = DQNPolicy(env_dim, action_dim)
        self.policy_network = DQNPolicy(env_dim, action_dim)
        self.policy_network.load_state_dict(self.target_network.state_dict())
        self.one_hot = F.one_hot(torch.arange(0, action_dim))
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.policy_network.parameters(), lr=LearningRate, momentum=0.9, weight_decay=1e-5)

    def enable_train(self):
        self.do_train = True
        self.cur_reward = 0.0
        self.steps = 0
        self.replay_buffer = []
    
    def reset_env(self):
        self.cur_reward = 0.0
        self.steps = 0

    @torch.no_grad()
    def select_action(self, state):
        if random.random() < Epsilon or len(self.replay_buffer) < BatchSize * 2:
            return random.choice(range(self.action_dim))
        state = torch.from_numpy(state)
        act_probs = self.target_network(state)
        return act_probs.argmax().item()

    def add_to_memory(self, state, action, reward, next_state, done):
        if len(self.replay_buffer) >= MaxBufferSize:
            self.replay_buffer.pop()
        reward = float(reward)
        self.replay_buffer.insert(0, Transition(torch.from_numpy(state), 
                                             torch.tensor([action]), 
                                             torch.from_numpy(next_state), 
                                             torch.tensor([reward]), done))

    def copy_new_params(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
    def learn_by_replay(self):
        if len(self.replay_buffer) <= BatchSize:
            return
        # Random choose BATCH_SIZE transitions.
        samples = random.sample(self.replay_buffer, BatchSize)
        states_batch = torch.stack([t.state for t in samples])
        next_states_batch = torch.stack([t.next_state for t in samples])
        actions_batch = torch.cat([t.action for t in samples]).to(torch.int64)
        rewards_batch = torch.stack([t.reward for t in samples])
        done_batch = torch.cat([torch.Tensor([t.done]) for t in samples])

        # Update policy_network
        with torch.no_grad():
            q_next = self.target_network(next_states_batch).max(1)[0].detach()
        act_vals = self.policy_network(states_batch).gather(dim=1, index=actions_batch.unsqueeze(1))
        label = rewards_batch.reshape(-1) + Gamma * q_next
        loss = self.criterion(act_vals, label.unsqueeze(1))
        print("loss ", loss)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()
