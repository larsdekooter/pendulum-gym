import gymnasium as gym
import torch

import numpy as np
from collections import deque
import random

env = gym.make("Pendulum-v1", render_mode="human")

observation, info = env.reset()
BATCHSIZE = 128

class DQN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stack = torch.nn.Sequential(torch.nn.Linear(3, 1))

    def forward(self, x):
        return self.stack(x)

class Network:
    def __init__(self) -> None:
        self.model = DQN()
        self.targetModel = DQN()
        self.targetModel.load_state_dict(self.model.state_dict())
        self.steps = 0
        self.memory = deque([], maxlen=100000)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.criterion = torch.nn.SmoothL1Loss()

    def getAction(self, state):
        epsilon = 0.1 + (1 - 0.1) * np.exp(-1e-5 * self.steps)
        self.steps += 1
        if np.random.random() < epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                return [self.model(torch.tensor(np.array([state]))).item()]

    def train(self, state, nextState, action, reward, done):
        self.memory.append((state, nextState, action, reward, done))
        if len(self.memory) < BATCHSIZE:
            return
        batch = random.sample(self.memory, BATCHSIZE)
        states, nextStates, actions, rewards, dones = (
            zip(*batch)
        )
        states = torch.tensor(states, dtype=torch.float32)
        nextStates = torch.tensor(nextStates, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.int64)

        qValues = self.model(states).gather(1, actions.unsqueeze(1)).sqeeuze(1)
        nextQValues = self.targetModel(nextStates).max(1)[0]

        targetQValues = rewards + (0.99 * nextQValues * (1-dones))
        
        loss = self.criterion(qValues, targetQValues.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



    def updateTarget(self):
        self.targetModel.load_state_dict(self.model.state_dict())

network = Network()
while True:
    ngames = 0
    action = network.getAction(observation)
    nextState, reward, terminated, truncated, info = env.step(action)

    network.train(observation, nextState, action, reward, terminated or truncated)
    observation = nextState
    if truncated or terminated:
        ngames += 1
        if ngames % 20 == 0:
            network.updateTarget()
        env.reset()
