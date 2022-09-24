from typing import Dict
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from module.schemas.metadata import AbstractActionValue
from variables.hyperparams.q_learning.c51 import HyperParams


class ActionValueModel(AbstractActionValue):
    def __init__(
        self,
        state_dim,
        action_dim,
        hyper_params: HyperParams,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hyper_params = hyper_params
        self.model = self.create_model()

    def create_model(self):
        input_state = Input((self.state_dim,))
        h1 = Dense(64, activation="relu")(input_state)
        h2 = Dense(64, activation="relu")(h1)
        outputs = []
        for _ in range(self.action_dim):
            outputs.append(
                Dense(self.hyper_params.ATOMS.value, activation="softmax")(h2)
            )
        return tf.keras.Model(input_state, outputs)

    def train(self, x, y):
        y = tf.stop_gradient(y)
        criterion = tf.keras.losses.CategoricalCrossentropy()
        opt = tf.keras.optimizers.Adam(self.hyper_params.LR.value)
        with tf.GradientTape() as tape:
            logits = self.model(x)
            loss = criterion(y, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def predict(self, state):
        return self.model.predict(state)

    def get_optimal_action(self, state):
        z = self.model.predict(state)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.hyper_params.Z.value)), axis=1)
        return np.argmax(q)

    def get_action(self, state, ep):
        state = np.reshape(state, [1, self.state_dim])
        eps = 1.0 / ((ep / 10) + 1)
        if np.random.rand() < eps:
            return np.random.randint(0, self.action_dim)
        else:
            return self.get_optimal_action(state)
