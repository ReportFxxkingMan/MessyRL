import numpy as np


class _TypedArray(np.ndarray):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        return np.array(val, dtype=cls.inner_type)


class _ArrayMeta(type):
    def __getitem__(self, t):
        return type("Array", (_TypedArray,), {"inner_type": t})


class AbstractArray(np.ndarray, metaclass=_ArrayMeta):
    pass
