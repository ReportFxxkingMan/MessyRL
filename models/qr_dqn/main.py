import tensorflow as tf
import gym
from models.qr_dqn.learn import Agent


input_dict = {
    "gamma" : 0.99,
    "batch_size" : 8,
    "lr" : 1e4,
    "atoms" : 8,
}

def main():
    env = gym.make("CartPole-v1")
    agent = Agent(env, input_dict)
    agent.train()


if __name__ == "__main__":
    main()
