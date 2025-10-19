"""
Script per avviare la UI/gioco interattivo (pygame).
Importa SnakeGame per rendering e Agent per giocare/visualizzare.
"""
import argparse
import pygame, sys
from env import SnakeGame
from dqn import Agent
from config import WIDTH, HEIGHT, FPS_VIS, FPS_TRAIN, CELL, WHITE, BLACK, RED, GREEN, GRAY

def run_play(args):
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
    losses = []

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
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        state = next_state
        if done:
            print(f"Episode {episode} ended. Score {env.score}")
            state = env.reset()
            episode += 1

        # Draw (reuse your drawing code or move into env.draw(screen,...))
        env.draw(screen, font, f"Mode: {'TRAIN' if training else 'PLAY' if greedy_play else 'IDLE'} | Ep: {episode} | Score: {env.score}")
        pygame.display.flip()
        clock.tick(FPS_TRAIN if training else FPS_VIS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run_play(args)