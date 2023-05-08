import pyjit
from typing import Any
from sensor import Sensor
from tensor import TensorXf
from film import Film
from point import Point2f, Point3f

integrators = {}


def register_integrator(name: str, init):
    integrators[name] = init


def new_integrator(desc: dict) -> "Integrator":
    return integrators[desc.pop("type")](desc.copy())


class Integrator:
    def __init__(self, desc: dict[str, Any]):
        ...

    def render(self, scene) -> TensorXf:
        from scene import Scene

        scene: Scene = scene

        sensor = scene.sensors[0]

        film = sensor.film()

        size = film.crop_size()

        wavefront_size = size[0] * size[1]

        sampler = sensor.sampler()

        # return res


class PathIntegrator(Integrator):
    def __init__(self, desc: dict):
        ...


register_integrator("path", lambda desc: PathIntegrator(desc))


if __name__ == "__main__":
    pyjit.set_backend("optix")
    film = Film(100, 100, 3)
    film.put(Point2f(10, 10), [1, 1, 1])

    import matplotlib.pyplot as plt

    film.tensor.schedule()
    pyjit.eval()

    plt.imshow(film.tensor.to_numpy())
    plt.show()
    plt.show()
    plt.show()
