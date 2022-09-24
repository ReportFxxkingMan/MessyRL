from enum import Enum
import gym
from models import QrDqnAgent
from variables.gamename import GameName
from variables.hyperparams.q_learning.qr_dqn import HyperParams


def train(
    game_name: str,
    agent,
    hyper_params: Enum,
):
    env = gym.make(game_name)
    agent = agent(env, hyper_params)
    agent.train()


if __name__ == "__main__":
    train(
        game_name=GameName.CARTPOLE_V1.value,
        agent=QrDqnAgent,
        hyper_params=HyperParams,
    )
