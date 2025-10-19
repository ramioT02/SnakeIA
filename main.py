import random
import math
from collections import deque, namedtuple
import numpy as np
import pygame
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


'''
T -->avvia/interrompe la simulazione
P --> gioca con rete allenata
S --> salva il modello in snake.dqn.pth
L --> carica il modello salvato in snake.dqn.pth
R --> reset 

'''

# -------------------------
#  Parametri di gioco
# -------------------------
WIDTH, HEIGHT = 600, 400
CELL = 20
COLS, ROWS = WIDTH // CELL, HEIGHT // CELL
FPS_VIS = 60          # refresh grafico
FPS_TRAIN = 600       # velocità durante training (più alto = più veloce)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WHITE=(255,255,255); BLACK=(0,0,0); RED=(200,0,0); GREEN=(0,200,0); GRAY=(60,60,60)

# -------------------------
#  Ambiente Snake (pygame)
# -------------------------
class SnakeGame:
    def __init__(self):
        self.reset()


    def reset(self):
        self.dir = (1,0)  # destra
        cx, cy = COLS//4, ROWS//2
        self.snake = [(cx, cy), (cx-1, cy), (cx-2, cy)]
        self.place_food()
        self.score = 0
        self.frame = 0
        return self.get_state()


    def place_food(self):
        positions = set((x,y) for x in range(COLS) for y in range(ROWS)) - set(self.snake)
        self.food = random.choice(list(positions))


    def step(self, action):
        # action: 0 = avanti, 1 = gira_destra, 2 = gira_sinistra
        self.frame += 1
        self.dir = self.new_direction(action)
        head = self.snake[0]
        new_head = (head[0]+self.dir[0], head[1]+self.dir[1])

        reward = 0.0
        done = False

        # muro 
        if (new_head[0] < 0 or new_head[0] >= COLS or
            new_head[1] < 0 or new_head[1] >= ROWS):
            done = True
            reward = -10.0
            return self.get_state(), reward, done, {}

        #se stesso
        if new_head in self.snake:
            done = True
            reward = -30.0
            return self.get_state(), reward, done, {}

        #movimento
        self.snake.insert(0, new_head)

        #controllo cibo
        if new_head == self.food:
            self.score += 1
            reward = 10.0 + 1.5 * (len(self.snake) - 3)
            self.place_food()
            self.frame = 0 #reset contatore stallo
        else:
            self.snake.pop()
            reward -= 2 #penalità se non mangia

        #ricompensa shaping per avvicinarsi/allontanarsi dal cibo
        dx_prev = abs(head[0]-self.food[0])+abs(head[1]-self.food[1])
        dx_now  = abs(new_head[0]-self.food[0])+abs(new_head[1]-self.food[1])
        if dx_now < dx_prev:
            reward += 3
        else:
            reward -= 1

        # anti-stallo: se gira per troppo tempo senza mangiare
        if self.frame > 100 + len(self.snake)*5:
            done = True
            reward -= 5.0

        return self.get_state(), reward, done, {}

    def new_direction(self, action):
        # ordine relativo in senso orario rispetto alla direzione corrente
        dx, dy = self.dir
        dirs = [(dx,dy), (dy,-dx), (-dy,dx)]  # avanti, destra, sinistra
        return dirs[action]

    def get_state(self):
        head = self.snake[0]
        dir_l = (self.dir[1], -self.dir[0])     # sinistra relativa
        dir_r = (-self.dir[1], self.dir[0])     # destra relativa
        front = (head[0]+self.dir[0], head[1]+self.dir[1])
        left  = (head[0]+dir_l[0],  head[1]+dir_l[1])
        right = (head[0]+dir_r[0],  head[1]+dir_r[1])

        def danger(cell):
            x,y = cell
            if x<0 or x>=COLS or y<0 or y>=ROWS: return 1.0
            return 1.0 if cell in self.snake else 0.0

        # feature (11):
        state = [
            danger(front), danger(right), danger(left),                  # 3
            float(self.dir==(0,-1)), float(self.dir==(0,1)),             # up, down
            float(self.dir==(-1,0)), float(self.dir==(1,0)),             # left, right
            float(self.food[1] < head[1]),  # food up
            float(self.food[1] > head[1]),  # food down
            float(self.food[0] < head[0]),  # food left
            float(self.food[0] > head[0]),  # food right
        ]
        return np.array(state, dtype=np.float32)

    # --- Rendering ---
    def draw(self, screen, font, info):
        screen.fill(BLACK)
        # griglia leggera
        for x in range(0, WIDTH, CELL):
            pygame.draw.line(screen, GRAY, (x,0), (x,HEIGHT))
        for y in range(0, HEIGHT, CELL):
            pygame.draw.line(screen, GRAY, (0,y), (WIDTH,y))

        # cibo
        fx, fy = self.food
        pygame.draw.rect(screen, RED, (fx*CELL, fy*CELL, CELL, CELL))
        # snake
        for i,(x,y) in enumerate(self.snake):
            rect = pygame.Rect(x*CELL, y*CELL, CELL, CELL)
            pygame.draw.rect(screen, GREEN if i==0 else (0,150,0), rect)

        # HUD
        txt = font.render(info, True, WHITE)
        screen.blit(txt, (10, 10))

# -------------------------
#  DQN minimale
# -------------------------
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
    def __init__(self, in_dim=11, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x): return self.net(x)

class Agent:
    def __init__(self, gamma=0.99, lr=1e-3, batch_size=128, eps_start=1.0, eps_end=0.05, eps_decay=6000, tau=0.01, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.online = DQN().to(self.device)
        self.target = DQN().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optim = optim.Adam(self.online.parameters(), lr=lr)
        self.memory = ReplayMemory()
        self.gamma=gamma
        self.batch_size=batch_size
        self.eps_start=eps_start
        self.eps_end=eps_end
        self.eps_decay=eps_decay
        self.tau=tau
        self.steps=0

    def select_action(self, state, greedy=False):
        self.steps += 1
        eps = 0.0 if greedy else self.eps_end + (self.eps_start-self.eps_end)*math.exp(-1.0*self.steps/self.eps_decay)
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

        # soft update
        for t,o in zip(self.target.parameters(), self.online.parameters()):
            t.data.copy_(t.data*(1-self.tau) + o.data*self.tau)

        return loss.item()

    def save(self, path='snake_dqn.pth'):
        torch.save(self.online.state_dict(), path)
    def load(self, path='snake_dqn.pth'):
        if os.path.exists(path):
            self.online.load_state_dict(torch.load(path, map_location=self.device))
            self.target.load_state_dict(self.online.state_dict())
            return True
        return False

# -------------------------
#  Loop principale pygame
# -------------------------
def main():

    # Dati per il grafico
    scores_history = []
    avg_scores = []
    loss_history = []

    plt.ion()  # modalità interattiva
    fig, ax = plt.subplots(2,1, figsize=(6,6))

    line1, = ax[0].plot([], [], label="Score")
    ax[0].set_title("Punteggio per episodio")
    ax[0].set_xlabel("Episodio")
    ax[0].set_ylabel("Score")
    ax[0].legend()
    line2, = ax[1].plot([], [], label="Loss")
    ax[1].set_title("Loss media")
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    plt.tight_layout()


    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake Sim")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 22)

    env = SnakeGame()
    agent = Agent()

    training = False
    greedy_play = False
    episode = 1
    best = 0
    losses = deque(maxlen=200)

    state = env.reset()

    while True:
        # Eventi
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                elif e.key == pygame.K_t:
                    training = not training
                    greedy_play = False
                elif e.key == pygame.K_p:
                    greedy_play = not greedy_play
                    training = False
                elif e.key == pygame.K_r:
                    state = env.reset()
                elif e.key == pygame.K_s:
                    agent.save(); print("Modello salvato.")
                elif e.key == pygame.K_l:
                    ok = agent.load(); print("Modello caricato." if ok else "File modello non trovato.")

        # Azione dalla policy
        action = agent.select_action(state, greedy=greedy_play)
        next_state, reward, done, _ = env.step(action)

        # Memorizza e allena (solo in training)
        if training:
            agent.store(state, action, next_state, reward, float(done))
            loss = agent.train_step()

            current_loss = loss if loss is not None else 0.0
            losses.append(current_loss)
            
            print(f"Step {agent.steps}, Memory size: {len(agent.memory)}, Loss: {loss}")

        state = next_state
        # Episodio finito
        if done:
            scores_history.append(env.score)
            
            if env.score > best:  # nuovo record
                best = env.score
                agent.save("snake_dqn.pth")  # salva automaticamente il modello migliore
                print(f"Nuovo best score {best}, modello salvato!")

            # Aggiorna info e reset episodio
            info_end = f"EP {episode} finito. Score: {env.score} | Best: {best}"
            print(info_end)
            state = env.reset()
            episode += 1

            avg_score = np.mean(scores_history[-100:])  # media ultimi 100 episodi
            avg_scores.append(avg_score)

            # Aggiorna la linea del punteggio
            line1.set_xdata(range(len(scores_history)))
            line1.set_ydata(scores_history)
            ax[0].relim()
            ax[0].autoscale_view()

            # Aggiorna la linea della loss media
            avg_loss = (sum(losses)/len(losses)) if losses else 0.0
            loss_history.append(avg_loss)
            line2.set_xdata(range(len(loss_history)))
            line2.set_ydata(loss_history)
            ax[1].relim()
            ax[1].autoscale_view()

            plt.pause(0.001)  # aggiorna il grafico senza bloccare

        # Render
        status = "TRAIN" if training else ("PLAY" if greedy_play else "IDLE")
        avg_loss = (sum(losses)/len(losses)) if losses else 0.0
        info = f"Mode: {status} | Ep: {episode} | Score: {env.score} | Best: {best} | Loss: {avg_loss:.4f}"
        env.draw(screen, font, info)
        pygame.display.flip()

        clock.tick(FPS_TRAIN if training else FPS_VIS)

if __name__ == "__main__":
    main()
