from typing import Any, Dict
import gym
from models.qr_dqn.learn import Agent


def qrdqn_main(
    game_name: str,
    params_dict: Dict[str, Any],
):
    env = gym.make(game_name)
    agent = Agent(env, params_dict)
    agent.train()
