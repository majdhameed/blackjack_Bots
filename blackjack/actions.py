# Shared action names used by the game engine, bots, and observations.
from enum import Enum


class Action(Enum):
    HIT = "hit"
    STAND = "stand"
    DOUBLE = "double"
    SPLIT = "split"
    SURRENDER = "surrender"

