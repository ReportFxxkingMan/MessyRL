from typing import Tuple
from collections import deque
import numpy as np
import random


class ReplayBuffer:
    '''
    Save the output of each step in an environment and provide an agent with batch samples.
    
    Parameters
    ----------
    batch_size : int 
        Batch size of an agent.
    capacity : int
        Max size of this buffer
    
    Returns
    -------
    None.
    
    
    '''
    def __init__(
        self,
        batch_size: int = 8,
        capacity: int = 10000,
    ):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def put(self, state, action, reward, next_state, done):
        '''
        Put a sample to this buffer.

        Parameters
        ----------
        state : array_like 
            state of this step in the environment.
        action : int or array_like
            action of this step in the environment.
        reward : TYPE
            reward of this step from the action.
        next_state : TYPE
            state of the next step from the action.
        done : 
            Boolean about the end of the environment.

        Returns
        -------
        None.

        '''
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self) -> Tuple[np.ndarray, ...]:
        sample = random.sample(self.buffer, min(len(self.buffer), self.batch_size))
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        # states = np.array(states).reshape(self.batch_size, -1)
        # next_states = np.array(next_states).reshape(self.batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)
    
    def clear_buffer(self):
        self.buffer = deque()
        