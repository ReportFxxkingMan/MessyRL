from typing import Dict
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape
from module.models.common.replaybuffer import ReplayBuffer
from module.schemas.metadata import AbstractActionValue
from module.schemas.common import Transition
from variables.hyperparams.q_learning.qr_dqn import HyperParams


class ActionValueModel(AbstractActionValue):
    def __init__(
        self, state_dim: int, action_dim: int, hyper_params: HyperParams
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hyper_params = hyper_params
        self.model = self.create_model()

    def create_model(self):
        input_state = Input((self.state_dim,))
        h1 = Dense(64, activation="relu")(input_state)
        h2 = Dense(64, activation="relu")(h1)
        outputs = Dense(
            self.action_dim * self.hyper_params.ATOMS.value, activation="linear"
        )(h2)
        reshaped_outputs = Reshape([self.action_dim, self.hyper_params.ATOMS.value])(
            outputs
        )
        return reshaped_outputs

    def quantile_huber_loss(self, target, pred, actions):
        pred = tf.reduce_sum(pred * tf.expand_dims(actions, -1), axis=1)
        pred_tile = tf.tile(
            tf.expand_dims(pred, axis=2), [1, 1, self.hyper_params.ATOMS.value]
        )
        target_tile = tf.tile(
            tf.expand_dims(target, axis=1), [1, self.hyper_params.ATOMS.value, 1]
        )

        _huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        huber_loss = _huber_loss(target_tile, pred_tile)

        _tau = tf.reshape(
            np.array(self.hyper_params.TAUS.value), [1, self.hyper_params.ATOMS.value]
        )
        _inv_tau = 1.0 - _tau

        tau = tf.tile(
            tf.expand_dims(_tau, axis=1), [1, self.hyper_params.ATOMS.value, 1]
        )
        inv_tau = tf.tile(
            tf.expand_dims(_inv_tau, axis=1), [1, self.hyper_params.ATOMS.value, 1]
        )
        error_loss = tf.math.subtract(target_tile, pred_tile)
        loss = tf.where(
            tf.less(error_loss, 0.0), inv_tau * huber_loss, tau * huber_loss
        )
        loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(loss, axis=2), axis=1))
        return loss

    def train(self, states, target, actions):
        with tf.GradientTape() as tape:
            theta = self.model(states)
            loss = self.quantile_huber_loss(target, theta, actions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        # self.opt = tf.keras.optimizers.Adam(params_dict["lr"])
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state, ep):
        state = np.reshape(state, [1, self.state_dim])
        eps = 1.0 / ((ep / 10) + 1)
        if np.random.rand() < eps:
            return np.random.randint(0, self.action_dim)
        else:
            return self.get_optimal_action(state)

    def get_optimal_action(self, state):
        z = self.model.predict(state)[0]
        q = np.mean(z, axis=1)
        return np.argmax(q)


class Agent:
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
