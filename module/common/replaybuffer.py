from typing import Tuple
import random
from collections import deque
import numpy as np
from module.common.metadata import AbstractBuffer
from module.schemas.common import Transition


class ReplayBuffer(AbstractBuffer):
    """
    Save the output of each step in an environment and provide an agent with batch samples.
    """

    def __init__(
        self,
        capacity: int = 10000,
    ) -> None:
        """
        Parameters
        Args:
            batch_size (int, optional): Batch size of an agent. Defaults to 8.
            capacity (int, optional): Max size of this buffer. Defaults to 10000.
        """
        self.buffer = deque(maxlen=capacity)
        self.buffer_count = 0

    def add(self, transition: Transition):
        if self.buffer_count < self.buffer_size:
            self.buffer.append(transition.to_tuple())
            self.buffer_count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(transition.to_tuple())

    def sample(self, batch_size: int) -> Tuple[np.ndarray]:
        sample = random.sample(self.buffer, min(self.buffer_count, batch_size))
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        return states, actions, rewards, next_states, done

    def count(self) -> int:
        return min(self.buffer_count, len(self.buffer))

    def clear_buffer(self) -> None:
        self.buffer = deque()
        self.count = 0
