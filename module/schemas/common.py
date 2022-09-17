from typing import Tuple
from pydantic import BaseModel


class Transition(BaseModel):
    state: float
    action: float
    reward: float
    next_state: float
    done: float

    def to_tuple(self) -> Tuple[float]:
        return (
            self.state,
            self.action,
            self.reward,
            self.next_state,
            self.done,
        )
