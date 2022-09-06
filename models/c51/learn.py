from typing import Dict
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from utils.replaybuffer import ReplayBuffer


class ActionValueModel:
    def __init__(
        self, 
        state_dim,
        action_dim,
        z, 
        input_dict:Dict
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.z = z
        
        self.atoms = input_dict["atoms"]
        self.opt = tf.keras.optimizers.Adam(input_dict["lr"])
        self.criterion = tf.keras.losses.CategoricalCrossentropy()
        self.model = self.create_model()

    def create_model(self):
        input_state = Input((self.state_dim,))
        h1 = Dense(64, activation="relu")(input_state)
        h2 = Dense(64, activation="relu")(h1)
        outputs = []
        for _ in range(self.action_dim):
            outputs.append(Dense(self.atoms, activation="softmax")(h2))
        return tf.keras.Model(input_state, outputs)

    def train(self, x, y):
        y = tf.stop_gradient(y)
        with tf.GradientTape() as tape:
            logits = self.model(x)
            loss = self.criterion(y, logits)
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
        z = self.model.predict(state)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        return np.argmax(q)


class Agent:
    def __init__(self, env, input_dict: Dict):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.buffer = ReplayBuffer()
        self.batch_size = input_dict["batch_size"]
        self.v_max = input_dict["v_max"]
        self.v_min = input_dict["v_min"]
        self.atoms = input_dict["atoms"]
        self.delta_z = float(self.v_max - self.v_min) / (self.atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.atoms)]
        self.gamma = input_dict["gamma"]
        self.q = ActionValueModel(self.state_dim, self.action_dim, self.z, input_dict)
        self.q_target = ActionValueModel(self.state_dim, self.action_dim, self.z, input_dict)
        self.target_update()

    def target_update(self):
        weights = self.q.model.get_weights()
        self.q_target.model.set_weights(weights)

    def replay(self):
        states, actions, rewards, next_states, dones = self.buffer.sample()
        z = self.q.predict(next_states)
        z_ = self.q_target.predict(next_states)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        q = q.reshape((self.batch_size, self.action_dim), order="F")
        next_actions = np.argmax(q, axis=1)
        m_prob = [
            np.zeros((self.batch_size, self.atoms)) for _ in range(self.action_dim)
        ]
        for i in range(self.batch_size):
            if dones[i]:
                Tz = min(self.v_max, max(self.v_min, rewards[i]))
                bj = (Tz - self.v_min) / self.delta_z
                l, u = math.floor(bj), math.ceil(bj)
                m_prob[actions[i]][i][int(l)] += u - bj
                m_prob[actions[i]][i][int(u)] += bj - l
            else:
                for j in range(self.atoms):
                    Tz = min(
                        self.v_max, max(self.v_min, rewards[i] + self.gamma * self.z[j])
                    )
                    bj = (Tz - self.v_min) / self.delta_z
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
                self.buffer.put(state, action, -1 if done else 0, next_state, done)

                if self.buffer.size() > 1000:
                    self.replay()
                if steps % 5 == 0:
                    self.target_update()

                state = next_state
                total_reward += reward
                steps += 1
            print("EP{} reward={}".format(ep, total_reward))