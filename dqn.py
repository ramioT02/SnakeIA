import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

Transition = namedtuple('Transition', ('state','action','next_state','reward','done'))

class ReplayMemory:
    def __init__(self, capacity=50000):
        self.buffer=deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch=random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    def __len__(self): return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, in_dim=11, out_dim=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

class Agent:
    def __init__(self, device=None, **kwargs):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.online = DQN().to(self.device)
        self.target = DQN().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optim = optim.Adam(self.online.parameters(), lr=kwargs.get("lr",1e-3))
        self.memory = ReplayMemory(capacity=kwargs.get("memory_capacity",50000))
        self.gamma=kwargs.get("gamma",0.99)
        self.batch_size=kwargs.get("batch_size",128)
        self.eps_start=kwargs.get("eps_start",1.0)
        self.eps_end=kwargs.get("eps_end",0.05)
        self.eps_decay=kwargs.get("eps_decay",6000)
        self.tau=kwargs.get("tau",0.01)
        self.steps=0

    def select_action(self, state, greedy=False):
        self.steps += 1
        eps = 0.0 if greedy else self.eps_end + (self.eps_start-self.eps_end)*np.exp(-1.0*self.steps/self.eps_decay)
        if random.random() < eps:
            return random.randrange(3)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(s)
            return int(q.argmax(1).item())

    def store(self, s,a,ns,r,d):
        self.memory.push(s, a, ns, r, d)

    def train_step(self):
        if len(self.memory) < self.batch_size: return None
        batch = self.memory.sample(self.batch_size)

        s = torch.tensor(batch.state, dtype=torch.float32, device=self.device)
        a = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        ns = torch.tensor(batch.next_state, dtype=torch.float32, device=self.device)
        r = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        d = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.online(s).gather(1, a)
        with torch.no_grad():
            max_next = self.target(ns).max(1, keepdim=True)[0]
            target = r + self.gamma * max_next * (1.0 - d)

        loss = nn.SmoothL1Loss()(q_values, target)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optim.step()

        for t,o in zip(self.target.parameters(), self.online.parameters()):
            t.data.copy_(t.data*(1-self.tau) + o.data*self.tau)

        return loss.item()

    def save(self, path='snake_dqn.pth'):
        torch.save(self.online.state_dict(), path)
    def load(self, path='snake_dqn.pth', map_location=None):
        if os.path.exists(path):
            self.online.load_state_dict(torch.load(path, map_location=self.device))
            self.target.load_state_dict(self.online.state_dict())
            return True
        return False