import random
from collections import deque

import numpy as np

from nn import MLP
from snake import N_ACTIONS


class Agent:
    def __init__(self, n_state, hidden=128, lr=1e-3, gamma=0.9, buffer_size=50_000, batch_size=64,
                 target_every=200, seed=0):
        self.net = MLP([n_state, hidden, N_ACTIONS], seed=seed)
        self.target = MLP([n_state, hidden, N_ACTIONS], seed=seed)
        self.target.copy_from(self.net)
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_every = target_every
        self.memory = deque(maxlen=buffer_size)
        self.updates = 0
        self.rng = random.Random(seed)

    def act(self, state, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.randrange(N_ACTIONS)
        q = self.net.forward(np.array([state], dtype=np.float64))[0]
        return int(np.argmax(q))

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.rng.sample(self.memory, self.batch_size)
        s, a, r, s2, done = zip(*batch)
        s = np.array(s, dtype=np.float64)
        s2 = np.array(s2, dtype=np.float64)
        a = np.array(a)
        r = np.array(r)
        done = np.array(done, dtype=np.float64)

        # target: reward + gamma * best q-value of the next state (from the slower target net)
        best_next = self.target.forward(s2).max(axis=1)
        target = r + self.gamma * best_next * (1 - done)

        q = self.net.forward(s)
        idx = np.arange(self.batch_size)
        # loss = mean of 0.5 * (q[a] - target)^2, only the action we took gets a gradient
        dout = np.zeros_like(q)
        dout[idx, a] = np.clip(q[idx, a] - target, -10, 10) / self.batch_size
        gW, gb = self.net.backward(dout)
        self.net.adam_step(gW, gb, self.lr)

        self.updates += 1
        if self.updates % self.target_every == 0:
            self.target.copy_from(self.net)