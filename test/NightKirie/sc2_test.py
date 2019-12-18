from enum import Enum
from types import SimpleNamespace
class Color(Enum):
    red = 1
    green = 2
    blue = 3

a = tuple(("1", "2", "3"))
b = {test: test for test in a}
b = SimpleNamespace(**b)
print(b)