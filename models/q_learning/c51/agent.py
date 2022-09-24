from typing import Dict
import math
import numpy as np
from models.q_learning.c51.action_value import ActionValueModel
from module.models.common.replaybuffer import ReplayBuffer
from module.schemas.common import Transition
from variables.hyperparams.q_learning.c51 import HyperParams


class Agent:
    def __init__(
        self,
        env,
        hyper_params: HyperParams,
    ) -> None:
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.hyper_params = hyper_params
        self.buffer = ReplayBuffer()
        self.q = ActionValueModel(
            self.state_dim,
            self.action_dim,
            self.hyper_params,
        )
        self.q_target = ActionValueModel(
            self.state_dim,
            self.action_dim,
            self.hyper_params,
        )
        self.target_update()

    def target_update(self):
        weights = self.q.model.get_weights()
        self.q_target.model.set_weights(weights)

    def replay(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(
            batch_size=self.hyper_params.BATCH_SIZE.value
        )
        z = self.q.predict(next_states)
        z_ = self.q_target.predict(next_states)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.hyper_params.Z.value)), axis=1)
        q = q.reshape((self.hyper_params.BATCH_SIZE.value, self.action_dim), order="F")
        next_actions = np.argmax(q, axis=1)
        m_prob = [
            np.zeros(
                (self.hyper_params.BATCH_SIZE.value, self.hyper_params.ATOMS.value)
            )
            for _ in range(self.action_dim)
        ]
        for i in range(self.hyper_params.BATCH_SIZE.value):
            if dones[i]:
                Tz = min(
                    self.hyper_params.V_MAX.value,
                    max(self.hyper_params.V_MIN.value, rewards[i]),
                )
                bj = (
                    Tz - self.hyper_params.V_MIN.value
                ) / self.hyper_params.DELTA_Z.value
                l, u = math.floor(bj), math.ceil(bj)
                m_prob[actions[i]][i][int(l)] += u - bj
                m_prob[actions[i]][i][int(u)] += bj - l
            else:
                for j in range(self.hyper_params.ATOMS.value):
                    Tz = min(
                        self.hyper_params.V_MAX.value,
                        max(
                            self.hyper_params.V_MIN.value,
                            rewards[i] + self.gamma * self.z[j],
                        ),
                    )
                    bj = (
                        Tz - self.hyper_params.V_MIN.value
                    ) / self.hyper_params.DELTA_Z.value
                    l, u = math.floor(bj), math.ceil(bj)
                    m_prob[actions[i]][i][int(l)] += z_[next_actions[i]][i][j] * (
                        u - bj
                    )
                    m_prob[actions[i]][i][int(u)] += z_[next_actions[i]][i][j] * (
                        bj - l
                    )
        self.q.train(states, m_prob)

    def train(self, max_epsiodes=500):
        for ep in range(max_epsiodes):
            done, total_reward, steps = False, 0, 0
            state = self.env.reset()
            while not done:
                action = self.q.get_action(state, ep)
                next_state, reward, done, _ = self.env.step(action)
                transition = Transition(
                    state=state,
                    action=action,
                    reward=-1 if done else 0,
                    next_state=next_state,
                    done=done,
                )
                self.buffer.add(transition=transition)

                if self.buffer.size() > 1000:
                    self.replay()
                if steps % 5 == 0:
                    self.target_update()

                state = next_state
                total_reward += reward
                steps += 1
            print("EP{} reward={}".format(ep, total_reward))
