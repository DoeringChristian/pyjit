import pyjit
from typing import Any
from sensor import Sensor
from tensor import TensorXf
from scene import Scene
from film import Film
from point import Point2f, Point3f


class Integrator:
    def __init__(self, desc: dict[str, Any]):
        ...

    def render(self, scene: Scene) -> TensorXf:
        sensor = scene.sensors[0]

        # return res


if __name__ == "__main__":
    pyjit.set_backend("optix")
    film = Film(100, 100, 3)
    film.put(Point2f(10, 10), [1, 1, 1])

    import matplotlib.pyplot as plt

    film.tensor.schedule()
    pyjit.eval()

    plt.imshow(film.tensor.to_numpy())
    plt.show()
