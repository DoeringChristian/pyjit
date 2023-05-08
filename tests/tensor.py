import pyjit
import numpy as np


class TensorXf:
    def __init__(self, val, shape: list[int]):
        size = 1
        for i in shape:
            size *= i

        self.data: pyjit.Var = pyjit.f32(float(val), num=size) + 0.0
        self.shape = shape

    def to_numpy(self):
        return self.data.to_numpy().reshape(self.shape)

    def schedule(self):
        self.data.schedule()


if __name__ == "__main__":
    pyjit.set_backend("optix")
    res = TensorXf(1.0, shape=[10])

    print(f"{res.data.size()=}")

    res.data.schedule()
    pyjit.eval()

    res = res.to_numpy()
    print(f"{res=}")
