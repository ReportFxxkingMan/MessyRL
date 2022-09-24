import numpy as np
from models.q_learning.qr_dqn.action_value import ActionValueModel
from module.common.replaybuffer import ReplayBuffer
from module.schemas.common import Transition
from variables.hyperparams.q_learning.qr_dqn import HyperParams
from module.schemas.metadata import AbstractAgent


class Agent(AbstractAgent):
    def __init__(self, env, hyper_params: HyperParams):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.buffer = ReplayBuffer()
        self.hyper_params = hyper_params
        self.q = ActionValueModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hyper_params=self.hyper_params,
        )
        self.q_target = ActionValueModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hyper_params=self.hyper_params,
        )
        self.target_update()

    def target_update(self):
        weights = self.q.model.get_weights()
        self.q_target.model.set_weights(weights)

    def replay(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(
            batch_size=self.hyper_params.BATCH_SIZE.value
        )
        q = self.q_target.predict(next_states)
        next_actions = np.argmax(np.mean(q, axis=2), axis=1)
        theta = []
        for i in range(self.hyper_params.BATCH_SIZE.value):
            if dones[i]:
                theta.append(np.ones(self.hyper_params.ATOMS.value) * rewards[i])
            else:
                theta.append(
                    rewards[i] + self.hyper_params.GAMMA.value * q[i][next_actions[i]]
                )
        self.q.train(states, theta, actions)

    def train(self, max_epsiodes=500):
        for ep in range(max_epsiodes):
            done, total_reward, steps = False, 0, 0
            state = self.env.reset()
            while not done:
                action = self.q.get_action(state, ep)
                next_state, reward, done, _ = self.env.step(action)
                action_ = np.zeros(self.action_dim)
                action_[action] = 1
                transition = Transition(
                    state=state,
                    action=action_,
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
