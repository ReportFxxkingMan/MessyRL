from enum import Enum
import gym
from models.q_learning.qr_dqn.agent import Agent
from variables.gamename import GameName
from variables.hyperparams.q_learning.qr_dqn import HyperParams


def train(
    game_name: str,
    hyper_params: Enum,
):
    env = gym.make(game_name)
    agent = Agent(env, hyper_params)
    agent.train()


if __name__ == "__main__":
    train(game_name=GameName.CARTPOLE_V1.value, hyper_params=HyperParams)
