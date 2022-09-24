from enum import Enum
import numpy as np


class HyperParams(Enum):
    GAMMA: float = 0.99
    BATCH_SIZE: int = 8
    LR: float = 1e-4
    ATOMS: int = 8
    TAUS: np.ndarray = np.array(
        [0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375],
        dtype=np.float32,
    )
