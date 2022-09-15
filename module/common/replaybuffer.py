from typing import Tuple
from collections import deque
import random
import numpy as np
from module.common.metadata import AbstractBuffer


class ReplayBuffer(AbstractBuffer):
    """
    Save the output of each step in an environment and provide an agent with batch samples.
    """

    def __init__(
        self,
        batch_size: int = 8,
        capacity: int = 10000,
    ) -> None:
        """
        Parameters
        Args:
            batch_size (int, optional): Batch size of an agent. Defaults to 8.
            capacity (int, optional): Max size of this buffer. Defaults to 10000.
        """
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self) -> Tuple[np.ndarray, ...]:
        sample = random.sample(self.buffer, min(len(self.buffer), self.batch_size))
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        return states, actions, rewards, next_states, done

    def size(self) -> int:
        return len(self.buffer)

    def clear_buffer(self) -> None:
        self.buffer = deque()
