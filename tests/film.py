import pyjit
from tensor import TensorXf
from point import Point2f

films = {}


def register_film(name: str, init):
    films[name] = init


def new_film(desc: dict):
    print(f"{desc=}")
    return films[desc.pop("type")](desc.copy())


class Film:
    tensor: TensorXf

    def __init__(self, desc: dict):
        self.width = desc.pop("width", 1024)
        self.height = desc.pop("height", 1024)
        self.n_channels = desc.pop("n_channels", 3)
        self.tensor = TensorXf(0.0, [self.height, self.width, self.n_channels])

    def put(self, pos: Point2f, spec: list[pyjit.Var], active=True):
        x = pyjit.u32(pos.x * self.width)
        y = pyjit.u32(pos.y * self.height)
        for i, s in enumerate(spec):
            pyjit.f32(s).scatter(
                self.tensor.data,
                (x + y * self.tensor.shape[1]) * self.tensor.shape[2] + i,
                mask=active,
            )

    def crop_size(self) -> tuple[int, int]:
        """
        Return width, height
        """
        return self.tensor.shape[1], self.tensor.shape[0]


register_film("hdrfilm", lambda desc: Film(desc))
