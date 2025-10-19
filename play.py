import argparse
import sys
import pygame

# Prova a importare le costanti da config.py; se non esiste, usa i default
try:
    from config import WIDTH, HEIGHT, CELL, FPS_VIS, FPS_TRAIN, WHITE, BLACK, RED, GREEN, GRAY
except Exception:
    WIDTH, HEIGHT = 600, 400
    CELL = 20
    FPS_VIS = 60
    FPS_TRAIN = 600
    WHITE=(255,255,255); BLACK=(0,0,0); RED=(200,0,0); GREEN=(0,200,0); GRAY=(60,60,60)

from env import SnakeGame
from dqn import Agent

def render(screen, font, info, env):
    # Sfondo
    screen.fill(BLACK)
    # Griglia
    for x in range(0, WIDTH, CELL):
        pygame.draw.line(screen, GRAY, (x,0), (x,HEIGHT))
    for y in range(0, HEIGHT, CELL):
        pygame.draw.line(screen, GRAY, (0,y), (WIDTH,y))
    # Cibo
    fx, fy = env.food
    pygame.draw.rect(screen, RED, (fx*CELL, fy*CELL, CELL, CELL))
    # Snake
    for i,(x,y) in enumerate(env.snake):
        rect = pygame.Rect(x*CELL, y*CELL, CELL, CELL)
        pygame.draw.rect(screen, GREEN if i==0 else (0,150,0), rect)
    # HUD
    txt = font.render(info, True, WHITE)
    screen.blit(txt, (10, 10))

def run_play():
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

    state = env.reset()
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                elif e.key == pygame.K_t:
                    training = not training; greedy_play = False
                elif e.key == pygame.K_p:
                    greedy_play = not greedy_play; training = False
                elif e.key == pygame.K_r:
                    state = env.reset()
                elif e.key == pygame.K_s:
                    agent.save(); print("Modello salvato.")
                elif e.key == pygame.K_l:
                    ok = agent.load(); print("Modello caricato." if ok else "File modello non trovato.")

        action = agent.select_action(state, greedy=greedy_play)
        next_state, reward, done, _ = env.step(action)

        if training:
            agent.store(state, action, next_state, reward, float(done))
            _ = agent.train_step()

        state = next_state
        if done:
            if env.score > best:
                best = env.score
                agent.save("snake_dqn.pth")
                print(f"Nuovo best score {best}, modello salvato!")
            print(f"Episode {episode} finito. Score: {env.score} | Best: {best}")
            state = env.reset()
            episode += 1

        status = "TRAIN" if training else ("PLAY" if greedy_play else "IDLE")
        info = f"Mode: {status} | Ep: {episode} | Score: {env.score} | Best: {best}"
        render(screen, font, info, env)
        pygame.display.flip()
        clock.tick(FPS_TRAIN if training else FPS_VIS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    run_play()