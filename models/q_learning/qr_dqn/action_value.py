import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape
from module.schemas.metadata import AbstractActionValue
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
        outputs = Dense(self.action_dim * self.hyper_params.ATOMS, activation="linear")(
            h2
        )
        reshaped_outputs = Reshape([self.action_dim, self.hyper_params.ATOMS])(outputs)
        return reshaped_outputs

    def quantile_huber_loss(self, target, pred, actions):
        pred = tf.reduce_sum(pred * tf.expand_dims(actions, -1), axis=1)
        pred_tile = tf.tile(
            tf.expand_dims(pred, axis=2), [1, 1, self.hyper_params.ATOMS]
        )
        target_tile = tf.tile(
            tf.expand_dims(target, axis=1), [1, self.hyper_params.ATOMS, 1]
        )

        _huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        huber_loss = _huber_loss(target_tile, pred_tile)

        _tau = tf.reshape(
            np.array(self.hyper_params.TAUS), [1, self.hyper_params.ATOMS]
        )
        _inv_tau = 1.0 - _tau

        tau = tf.tile(tf.expand_dims(_tau, axis=1), [1, self.hyper_params.ATOMS, 1])
        inv_tau = tf.tile(
            tf.expand_dims(_inv_tau, axis=1), [1, self.hyper_params.ATOMS, 1]
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
