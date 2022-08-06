# DQN load and play
# coded by St.Watermelon

import gym
import numpy as np
import tensorflow as tf
from dqn_learn import DQNagent


def main():

    env_name = "CartPole-v1"
    env = gym.make(env_name)

    print(env.observation_space.shape[0])  # 4
    # get action dimension
    print(env.action_space, env.observation_space)

    agent = DQNagent(env)

    agent.load_weights("./save_weights/")

    time = 0
    state = env.reset()

    while True:
        env.render()
        qs = agent.dqn(tf.convert_to_tensor([state], dtype=tf.float32))
        action = np.argmax(qs.numpy())

        state, reward, done, _ = env.step(action)
        time += 1

        print("Time: ", time, "Reward: ", reward)

        if done:
            break

    env.close()


if __name__ == "__main__":
    main()
