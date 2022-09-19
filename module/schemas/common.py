from typing import Tuple
from pydantic import BaseModel
import numpy as np


class Transition(BaseModel):
    state: np.ndarray
    action: np.ndarray
    reward: int
    next_state: np.ndarray
    done: bool

    class Config:
        arbitrary_types_allowed = True

    def to_tuple(self) -> Tuple[float]:
        return (
            self.state,
            self.action,
            self.reward,
            self.next_state,
            self.done,
        )
