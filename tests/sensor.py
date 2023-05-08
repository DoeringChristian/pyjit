import pyjit
from point import Point2f

sensors = {}


def register_sensor(name: str, init):
    sensors[name] = init


class Sensor:
    def __init__(self, desc: dict):
        ...

    def sample_ray(
        time: pyjit.Var, sample1: pyjit.Var, sample2: Point2f, sample3: Point2f
    ):
        ...
