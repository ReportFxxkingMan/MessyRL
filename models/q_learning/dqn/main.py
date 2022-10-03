import gym
from models.q_learning.dqn.learn import DQNagent


def main():
    max_episode_num = 500
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    agent = DQNagent(env)

    agent.train(max_episode_num)

    agent.plot_result()


if __name__ == "__main__":
    main()
