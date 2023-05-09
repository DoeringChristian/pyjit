import pyjit
from typing import Any
from sensor import Sensor
from tensor import TensorXf
from film import Film
from point import Point2f, Point3f
from vector import Vector2f

integrators = {}


def register_integrator(name: str, init):
    integrators[name] = init


def new_integrator(desc: dict) -> "Integrator":
    return integrators[desc.pop("type")](desc.copy())


class Integrator:
    def __init__(self, desc: dict[str, Any]):
        ...

    def render(self, scene, seed=0) -> TensorXf:
        ...


class PathIntegrator(Integrator):
    def __init__(self, desc: dict):
        super().__init__(desc)

    def render(self, scene, seed=0) -> TensorXf:
        from scene import Scene

        scene: Scene = scene

        sensor = scene.sensors[0]

        film = sensor.film()

        size = film.crop_size()

        wavefront_size = size[0] * size[1]

        sampler = sensor.sampler()
        sampler.seed(seed, wavefront_size)

        idx = pyjit.index(wavefront_size)
        pos = Point2f(0.0, 0.0)
        pos.y = idx.div(pyjit.f32(size[0]))
        pos.x = pos.y.fma(pyjit.f32(-size[0]), idx)

        offset = sampler.next_2d()
        sample_pos = Point2f(
            (pos.x + offset.x).div(pyjit.f32(size[0])),
            (pos.y + offset.x).div(pyjit.f32(size[1])),
        )

        ray = sensor.sample_ray(0.0, 0.0, pos, Point2f(0, 0))

        pi = scene.intersect_preliminary(ray)

        film.put(pos, [pi.uv.x, pi.uv.y, 0.00])

        pyjit.eval()

        return film.tensor


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
