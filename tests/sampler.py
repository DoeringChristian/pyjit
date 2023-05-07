import pyjit
from typing import Any


class Sampler:
    def __init__(self, desc: dict[str, Any]):
        self.sampl_count = desc.get("sample_count", default=4)
        self.base_seed = desc.get("seed", default=0)

        self.dimension_index = pyjit.u32(0)
        self.sample_index = pyjit.u32(0)
        self.samples_per_wavefront = 1
        self.wavefront_size = 0

    def seed(self, seed: int, wavefront_size: int | None):
        if wavefront_size is not None:
            self.wavefront_size = wavefront_size

        ...
