from typing import Tuple
from pydantic import BaseModel
import numpy as np
from module.schemas.type import AbstractArray


class Transition(BaseModel):
    state: AbstractArray[np.float32]
    action: AbstractArray[np.float32]
    reward: float
    next_state: AbstractArray[np.float32]
    done: bool

    def to_tuple(self) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        return (
            self.state,
            self.action,
            self.reward,
            self.next_state,
            self.done,
        )
