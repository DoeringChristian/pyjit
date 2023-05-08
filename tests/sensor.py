import pyjit
from point import Point2f
from transform import Matrix4f
from film import Film, new_film
from sampler import Sampler, new_sampler

sensors = {}


def register_sensor(name: str, init):
    sensors[name] = init


def new_sensor(desc: dict) -> "Sensor":
    return sensors[desc.pop("type")](desc.copy())


class Sensor:
    def __init__(self, desc: dict):
        self.__sampler = new_sampler(desc.pop("sampler", {"type": "independent"}))
        self.__film = new_film(desc.pop("film", {"type": "hdrfilm"}))

    def sample_ray(
        self, time: pyjit.Var, sample1: pyjit.Var, sample2: Point2f, sample3: Point2f
    ):
        ...

    def film(self) -> Film:
        return self.__film

    def sampler(self) -> Sampler:
        return self.__sampler


class Orthogonal(Sensor):
    def __init__(self, desc):
        super().__init__(desc)
        self.to_world = Matrix4f(desc.pop("to_world"))

    def sample_ray(
        self, time: pyjit.Var, sample1: pyjit.Var, sample2: Point2f, sample3: Point2f
    ):
        ...


register_sensor("orthogonal", lambda desc: Orthogonal(desc))
