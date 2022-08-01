import pytest
from models.example import add, subtract


def test_add():
    assert add(1, 2) == 3
    assert add(2.5, 1.5) == 4.0


def test_subtract():
    assert subtract(1, 2) == -1
    assert subtract(3, 1) == 2
