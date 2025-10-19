"""
Script di training headless.
Esegue episodi senza pygame: utile per training veloce e CI.
"""
import argparse
import os
from collections import deque
import numpy as np

from env import SnakeGame
from dqn import Agent
from utils import set_seed, save_checkpoint, plot_metrics

def run_training(args):
    set_seed(args.seed)
    env = SnakeGame()
    agent = Agent(device=args.device)

    scores_history = []
    losses = deque(maxlen=200)
    episode = 1
    best = -1

    while episode <= args.episodes:
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, greedy=False)
            ns, r, done, _ = env.step(action)
            agent.store(state, action, ns, r, float(done))
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            state = ns

        scores_history.append(env.score)
        avg_loss = np.mean(losses) if losses else 0.0

        if env.score > best:
            best = env.score
            save_checkpoint(agent, os.path.join(args.outdir, "best.pth"), meta={"episode": episode, "score": best})

        if episode % args.log_every == 0:
            print(f"Episode {episode}, score {env.score}, best {best}, avg_loss {avg_loss:.4f}")

        episode += 1

    plot_metrics(scores_history, list(losses), fname=os.path.join(args.outdir, "metrics.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="runs")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    run_training(args)