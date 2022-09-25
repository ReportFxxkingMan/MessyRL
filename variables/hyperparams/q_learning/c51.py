import numpy as np
from module.schemas.metadata import AbstractHyperParams


class HyperParams(AbstractHyperParams):
    GAMMA: float = 0.99
    BATCH_SIZE: int = 8
    LR: float = 1e-4
    ATOMS: int = 8
    V_MIN: float = -5.0
    V_MAX: float = 5.0
    DELTA_Z: float = 10 / 7
    Z: np.ndarray = np.array(
        [-5 + i * 10 / 7 for i in range(8)],
        dtype=np.float32,
    )
