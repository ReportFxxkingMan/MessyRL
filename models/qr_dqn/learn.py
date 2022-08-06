from typing import Dict
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape
from models.qr_dqn.replaybuffer import ReplayBuffer


class ActionValueModel:
    def __init__(self, state_dim, action_dim, input_dict: Dict):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atoms = input_dict["atoms"]
        self.tau = [
            (2 * (i - 1) + 1) / (2 * self.atoms) for i in range(1, self.atoms + 1)
        ]
        self.huber_loss = tf.keras.losses.Huber(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.opt = tf.keras.optimizers.Adam(input_dict["lr"])
        self.model = self.create_model()

    def create_model(self):
        return tf.keras.Sequential(
            [
                Input(
                    [
                        self.state_dim,
                    ]
                ),
                Dense(64, activation="relu"),
                Dense(64, activation="relu"),
                Dense(self.action_dim * self.atoms, activation="linear"),
                Reshape([self.action_dim, self.atoms]),
            ]
        )

    def quantile_huber_loss(self, target, pred, actions):
        pred = tf.reduce_sum(pred * tf.expand_dims(actions, -1), axis=1)
        pred_tile = tf.tile(tf.expand_dims(pred, axis=2), [1, 1, self.atoms])
        target_tile = tf.tile(tf.expand_dims(target, axis=1), [1, self.atoms, 1])
        huber_loss = self.huber_loss(target_tile, pred_tile)
        tau = tf.reshape(np.array(self.tau), [1, self.atoms])
        inv_tau = 1.0 - tau
        tau = tf.tile(tf.expand_dims(tau, axis=1), [1, self.atoms, 1])
        inv_tau = tf.tile(tf.expand_dims(inv_tau, axis=1), [1, self.atoms, 1])
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
    def __init__(self, env, input_dict: Dict):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.buffer = ReplayBuffer()
        self.batch_size = input_dict["batch_size"]
        self.atoms = input_dict["atoms"]
        self.gamma = input_dict["gamma"]
        self.q = ActionValueModel(self.state_dim, self.action_dim)
        self.q_target = ActionValueModel(self.state_dim, self.action_dim)
        self.target_update()

    def target_update(self):
        weights = self.q.model.get_weights()
        self.q_target.model.set_weights(weights)

    def replay(self):
        states, actions, rewards, next_states, dones = self.buffer.sample()
        q = self.q_target.predict(next_states)
        next_actions = np.argmax(np.mean(q, axis=2), axis=1)
        theta = []
        for i in range(self.batch_size):
            if dones[i]:
                theta.append(np.ones(self.atoms) * rewards[i])
            else:
                theta.append(rewards[i] + self.gamma * q[i][next_actions[i]])
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
                self.buffer.put(state, action_, -1 if done else 0, next_state, done)

                if self.buffer.size() > 1000:
                    self.replay()
                if steps % 5 == 0:
                    self.target_update()

                state = next_state
                total_reward += reward
                steps += 1
            print("EP{} reward={}".format(ep, total_reward))