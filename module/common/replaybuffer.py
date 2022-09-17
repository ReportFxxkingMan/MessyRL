from typing import Tuple
import random
from collections import deque
import numpy as np
from module.common.metadata import AbstractBuffer
from module.schemas.common import Transition


class ReplayBuffer(AbstractBuffer):
    """
    Save the output of each step in an environment and provide an agent with batch samples.
    Methods:
        add: add buffer
        sample: sample buffer
        count: count buffer
        clear_buffer: clear buffer
    """

    def __init__(
        self,
        capacity: int = 10000,
    ) -> None:
        """
        Parameters
        Args:
            capacity (int, optional): Max size of this buffer. Defaults to 10000.
        """
        self.buffer = deque(maxlen=capacity)
        self.buffer_count = 0

    def add(self, transition: Transition) -> None:
        """
        Add transition to buffer
        Args:
            transition (Transition):
        """
        if self.buffer_count < self.buffer_size:
            self.buffer.append(transition.to_tuple())
            self.buffer_count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(transition.to_tuple())

    def sample(self, batch_size: int) -> Tuple[np.ndarray]:
        """
        Sampling buffer
        Args:
            batch_size (int): sample size
        Returns:
            Tuple[np.ndarray]: states, actions, rewards, next_states, done
        """
        sample = random.sample(self.buffer, min(self.buffer_count, batch_size))
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        return states, actions, rewards, next_states, done

    def count(self) -> int:
        """
        Count of buffer
        Returns:
            int: min(buffer_count, buffer)
        """
        return min(self.buffer_count, len(self.buffer))

    def clear(self) -> None:
        """
        Clear buffer
        """
        self.buffer = deque()
        self.buffer_count = 0
