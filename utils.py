import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_checkpoint(agent, path, meta=None):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    payload = {
        "state_dict": agent.online.state_dict(),
        "meta": meta or {}
    }
    torch.save(payload, path)

def load_checkpoint(agent, path, map_location=None):
    if not os.path.exists(path): return False
    payload = torch.load(path, map_location=map_location)
    agent.online.load_state_dict(payload["state_dict"])
    agent.target.load_state_dict(agent.online.state_dict())
    return True

def plot_metrics(scores, losses, fname=None):
    fig, ax = plt.subplots(2,1, figsize=(6,6))
    ax[0].plot(scores); ax[0].set_title("Scores")
    ax[1].plot(losses); ax[1].set_title("Loss")
    plt.tight_layout()
    if fname:
        fig.savefig(fname)
    else:
        plt.show()
    plt.close(fig)