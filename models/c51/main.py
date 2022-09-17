import gym
from models.c51.learn import Agent


def main():
    env = gym.make("CartPole-v1")
    agent = Agent(
        env,
        input_dict={
            "gamma": 0.99,
            "lr": 1e4,
            "batch_size": 8,
            "atoms": 8,
            "v_min": -5.0,
            "v_max": 5.0,
        },
    )
    agent.train()


if __name__ == "__main__":
    main()
