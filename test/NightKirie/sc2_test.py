from enum import Enum
from types import SimpleNamespace
class Color(Enum):
    red = 1
    green = 2
    blue = 3

a = (tuple(["1", "2", "3"]) + 
    tuple(["4", "5", "6"]))
print(a)