import pyjit
from typing import Any
from rand import PCG32, sample_tea_32
from point import Point2f

samplers = {}


def register_sampler(name: str, init):
    samplers[name] = init


def new_sampler(desc: dict):
    return samplers[desc.pop("type")](desc.copy())


class Sampler:
    def __init__(self, desc: dict[str, Any]):
        self.sampl_count = desc.pop("sample_count", 4)
        self.base_seed = desc.pop("seed", 0)

        self.dimension_index = pyjit.u32(0)
        self.sample_index = pyjit.u32(0)
        self.samples_per_wavefront = 1
        self.wavefront_size = 0

    def seed(self, seed: int, wavefront_size: int | None):
        if wavefront_size is not None:
            self.wavefront_size = wavefront_size
        self.dimension_index = pyjit.u32(0)
        self.sample_index = pyjit.u32(0)

    def advance(self):
        self.dimension_index = pyjit.u32(0)
        self.sample_index += 1

    def next_1d(self) -> pyjit.Var:
        ...

    def next_2d(self) -> Point2f:
        ...

    def sample_count(self) -> int:
        ...

    def set_sample_count(self, spp: int):
        ...

    def schedule(self):
        ...


class Independent(Sampler):
    def __init__(self, desc):
        super().__init__(desc)

    def seed(self, seed: int, wavefront_size: int | None):
        super().seed(seed, wavefront_size)

        seed_value = self.base_seed + seed

        idx = pyjit.index(wavefront_size)
        tmp = pyjit.u32(seed_value)

        v0, v1 = sample_tea_32(tmp, idx)

        self.rng = PCG32(1, v0, v1)

    def schedule(self):
        self.rng.inc.schedule()
        self.rng.state.schedule()

    def next_1d(self):
        return self.rng.next_f32()

    def next_2d(self) -> Point2f:
        return Point2f(self.rng.next_f32(), self.rng.next_f32())


register_sampler("independent", lambda desc: Independent(desc))

if __name__ == "__main__":
    pyjit.set_backend("optix")
    sampler = Independent({})

    sampler.seed(0, 100)

    p = sampler.next_2d()

    print(f"{sampler.next_2d()=}")
