import argparse
import random
import time

import numpy as np

from agent import Agent
from snake import Snake

parser = argparse.ArgumentParser()
parser.add_argument("--games", type=int, default=1500)
parser.add_argument("--size", type=int, default=10)
parser.add_argument("--features", default="basic", choices=["basic", "space"])
parser.add_argument("--tag", default="basic", help="name used for the files in results/")
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--eps_games", type=int, default=600, help="games over which epsilon goes from 1 to 0.01")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
env = Snake(args.size, seed=args.seed, features=args.features)
agent = Agent(env.n_state, hidden=args.hidden, lr=args.lr, gamma=args.gamma, seed=args.seed)

log = open(f"results/{args.tag}_log.txt", "w", encoding="utf-8")


def say(msg):
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


def play(agent, env, games, epsilon):
    scores = []
    for _ in range(games):
        s = env.reset()
        while not env.done:
            s, _, _ = env.step(agent.act(s, epsilon))
        scores.append(env.score)
    return scores


say(str(vars(args)))

# baseline: an agent that only moves randomly
random_scores = play(agent, env, 200, epsilon=1.0)
np.savetxt("results/random_scores.txt", random_scores, fmt="%d")
say(f"random agent, 200 games: mean score {np.mean(random_scores):.2f}, best {max(random_scores)}")

scores = []
best = 0
t0 = time.time()
for game in range(args.games):
    epsilon = max(0.01, 1 - game / args.eps_games)
    s = env.reset()
    while not env.done:
        a = agent.act(s, epsilon)
        s2, r, done = env.step(a)
        agent.remember(s, a, r, s2, done)
        agent.learn()
        s = s2
    scores.append(env.score)
    best = max(best, env.score)
    if (game + 1) % 50 == 0:
        say(f"game {game + 1}: mean score of last 50 = {np.mean(scores[-50:]):.2f}, "
            f"best {best}, epsilon {epsilon:.2f} | {(time.time() - t0) / 60:.1f} min")

np.savetxt(f"results/{args.tag}_train_scores.txt", scores, fmt="%d")
agent.net.save(f"results/{args.tag}_net.npz")

# final test: no random moves at all
test_scores = play(agent, env, 100, epsilon=0.0)
np.savetxt(f"results/{args.tag}_test_scores.txt", test_scores, fmt="%d")
say(f"trained agent, 100 games without exploration: mean score {np.mean(test_scores):.2f}, best {max(test_scores)}")
say(f"training time: {(time.time() - t0) / 60:.1f} min")