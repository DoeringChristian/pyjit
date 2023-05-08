import pyjit
from tensor import TensorXf
from point import Point2f

films = {}


def register_film(name: str, init):
    films[name] = init


def new_film(desc: dict):
    return films[desc.pop("type")](desc)


class Film:
    tensor: TensorXf

    def __init__(self, width: int, height: int, n_channels):
        self.tensor = TensorXf(0.0, [height, width, n_channels])

    def put(self, pos: Point2f, spec: list[pyjit.Var], active=True):
        for i, s in enumerate(spec):
            pyjit.f32(s).scatter(
                self.tensor.data,
                (pos.x + pos.y * self.tensor.shape[1]) * self.tensor.shape[2] + i,
                mask=active,
            )

    def crop_size(self) -> (int, int):
        """
        Return width, height
        """
        return self.tensor.shape[1], self.tensor.shape[0]
