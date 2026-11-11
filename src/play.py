import argparse

import numpy as np
from PIL import Image

from agent import Agent
from snake import Snake

parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="space")
parser.add_argument("--features", default="space", choices=["basic", "space"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--hidden", type=int, default=128)
args = parser.parse_args()

env = Snake(10, seed=args.seed, features=args.features)
agent = Agent(env.n_state, hidden=args.hidden)
agent.net.load(f"results/{args.tag}_net.npz")

CELL = 24
COLORS = {"bg": (242, 242, 242), "body": (27, 175, 122), "head": (14, 110, 76), "food": (214, 64, 48)}


def frame():
    img = np.zeros((env.size, env.size, 3), dtype=np.uint8)
    img[:, :] = COLORS["bg"]
    img[env.food[1], env.food[0]] = COLORS["food"]
    for x, y in env.body[1:]:
        img[y, x] = COLORS["body"]
    hx, hy = env.body[0]
    img[hy, hx] = COLORS["head"]
    return Image.fromarray(np.kron(img, np.ones((CELL, CELL, 1), dtype=np.uint8)))


s = env.reset()
frames = [frame()]
while not env.done:
    s, _, _ = env.step(agent.act(s, 0.0))
    frames.append(frame())

step = max(1, len(frames) // 400)  # keep the gif small
frames = frames[::step]
frames[0].save(f"results/play_{args.tag}.gif", save_all=True, append_images=frames[1:], duration=70, loop=0)
print(f"score {env.score}, {len(frames)} frames saved to results/play_{args.tag}.gif")