from typing import Tuple
from collections import deque
import numpy as np
import random


class ReplayBuffer:
    def __init__(
        self, 
        batch_size:int=8,
        capacity:int=10000,
    ):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def put(
        self, 
        state, 
        action, 
        reward, 
        next_state, 
        done
    ):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(
        self
    ) -> Tuple[np.ndarray, ...]:
        sample = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(self.batch_size, -1)
        next_states = np.array(next_states).reshape(self.batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)
