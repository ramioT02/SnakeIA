import random
import numpy as np

from config import COLS, ROWS

class SnakeGame:
    """
    Logica dell'ambiente Snake (senza pygame).
    Metodi: reset(), step(action), get_state(), place_food(), new_direction(action)
    action: 0 avanti, 1 gira destra, 2 gira sinistra
    """
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.reset()

    def reset(self):
        self.dir = (1,0)
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
        self.frame += 1
        self.dir = self.new_direction(action)
        head = self.snake[0]
        new_head = (head[0]+self.dir[0], head[1]+self.dir[1])

        reward = 0.0
        done = False

        # muro
        if (new_head[0] < 0 or new_head[0] >= COLS or
            new_head[1] < 0 or new_head[1] >= ROWS):
            return self.get_state(), -10.0, True, {}

        # se stesso
        if new_head in self.snake:
            return self.get_state(), -30.0, True, {}

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            reward = 10.0 + 1.5 * (len(self.snake) - 3)
            self.place_food()
            self.frame = 0
        else:
            self.snake.pop()
            reward -= 2

        # shaping verso il cibo
        dx_prev = abs(head[0]-self.food[0])+abs(head[1]-self.food[1])
        dx_now  = abs(new_head[0]-self.food[0])+abs(new_head[1]-self.food[1])
        reward += 3 if dx_now < dx_prev else -1

        if self.frame > 100 + len(self.snake)*5:
            return self.get_state(), reward-5.0, True, {}

        return self.get_state(), reward, done, {}

    def new_direction(self, action):
        dx, dy = self.dir
        dirs = [(dx,dy), (dy,-dx), (-dy,dx)]
        return dirs[action]

    def get_state(self):
        head = self.snake[0]
        dir_l = (self.dir[1], -self.dir[0])
        dir_r = (-self.dir[1], self.dir[0])
        front = (head[0]+self.dir[0], head[1]+self.dir[1])
        left  = (head[0]+dir_l[0],  head[1]+dir_l[1])
        right = (head[0]+dir_r[0],  head[1]+dir_r[1])

        def danger(cell):
            x,y = cell
            if x<0 or x>=COLS or y<0 or y>=ROWS: return 1.0
            return 1.0 if cell in self.snake else 0.0

        state = [
            danger(front), danger(right), danger(left),
            float(self.dir==(0,-1)), float(self.dir==(0,1)),
            float(self.dir==(-1,0)), float(self.dir==(1,0)),
            float(self.food[1] < head[1]),
            float(self.food[1] > head[1]),
            float(self.food[0] < head[0]),
            float(self.food[0] > head[0]),
        ]
        return np.array(state, dtype=np.float32)