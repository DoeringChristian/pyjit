import pyjit
from point import Point2f
from film import Film, new_film
from sampler import Sampler, new_sampler

sensors = {}


def register_sensor(name: str, init):
    sensors[name] = init


class Sensor:
    def __init__(self, desc: dict):
        self.__sampler = new_sampler(desc.pop("sampler"))
        self.__film = new_film(desc.pop("film"))

    def sample_ray(
        time: pyjit.Var, sample1: pyjit.Var, sample2: Point2f, sample3: Point2f
    ):
        ...

    def film(self) -> Film:
        ...

    def sampler(self) -> Sampler:
        return self.__sampler


class Orthogonal(Sensor):
    def __init__(self, desc):
        super().__init__(desc)
        ...
