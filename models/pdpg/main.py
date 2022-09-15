import gym
from models.pdpg.learn import PDPGagent
import argparse

parser = argparse.ArgumentParser(description="hyperparams")
parser.add_argument("--epochs", required=False, default=10000, type=int)
parser.add_argument("--lr1", required=False, default=0.00002, type=float)
parser.add_argument("--lr2", required=False, default=0.00002, type=float)
parser.add_argument("--bs", required=False, default=128, type=int)
parser.add_argument("--bfs", required=False, default=1000000, type=int)
parser.add_argument("--gamma", required=False, default=0.9, type=float)
parser.add_argument("--delta", required=False, default=0.2, type=float)
parser.add_argument("--qs", required=False, default=24, type=int)
parser.add_argument("--tau", required=False, default=0.01, type=float)

args = parser.parse_args()


def main():
    max_episode_num = args.epochs
    env_name = "BipedalWalker-v3"
    env = gym.make(env_name)
    agent = PDPGagent(
        env,
        args.gamma,
        args.bs,
        args.bfs,
        args.lr1,
        args.lr2,
        args.delta,
        args.qs,
        args.tau,
    )

    agent.train(max_episode_num)

    agent.plot_result()


if __name__ == "__main__":
    main()
