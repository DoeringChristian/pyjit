import pyjit
import mitsuba as mi
import drjit as dr


def sample_tea_32(
    v0: pyjit.Var, v1: pyjit.Var, rounds=4
) -> tuple[pyjit.Var, pyjit.Var]:
    sum: pyjit.Var = pyjit.u32(0)
    for i in range(rounds):
        sum += 0x9E3779B9
        v0 += (v1.shl(4) + 0xA341316C) ^ (v1 + sum) ^ (v1.shr(5) + 0xC8013EA4)
        v1 += (v0.shl(4) + 0xAD90777D) ^ (v0 + sum) ^ (v0.shr(5) + 0x7E95761E)

    return v0, v1


def sample_tea_64(v0: pyjit.Var, v1: pyjit.Var, rounds=4) -> pyjit.Var:
    v0, v1 = sample_tea_32(v0, v1, rounds)
    return pyjit.u64(v0) + pyjit.u64(v1).shl(32)


PCG32_DEFAULT_STATE = 0x853C49E6748FEA9B
PCG32_DEFAULT_STREAM = 0xDA3E39CB94B95BDB
PCG32_MULT = 0x5851F42D4C957F2D


class PCG32:
    def __init__(
        self, size=1, initstate=PCG32_DEFAULT_STATE, initseq=PCG32_DEFAULT_STREAM
    ):
        self.state = pyjit.u64(0)
        self.inc = (pyjit.u64(initseq) + pyjit.u64(pyjit.index(size))).shl(1) | 1
        self.next_u32()
        self.state += initstate
        self.next_u32()

    def next_u32(self) -> pyjit.Var:
        oldstate = self.state

        self.state: pyjit.Var = oldstate.fma(pyjit.u64(PCG32_MULT), self.inc)

        xorshift = pyjit.u32((oldstate.shr(18) ^ oldstate).shr(27))
        rot = oldstate.shr(59)

        return xorshift.shr(rot) | (xorshift.shl(pyjit.i32(rot).neg() & 31))

    def next_u64(self) -> pyjit.Var:
        v0 = self.next_u32()
        v1 = self.next_u32()

        return pyjit.u64(v0) | pyjit.u64(v1).shl(32)

    def next_f32(self) -> pyjit.Var:
        return (self.next_u32().shr(9) | 0x3F800000).bitcast("f32") - 1.0

    def next_f64(self) -> pyjit.Var:
        return (self.next_u64().shr(9) | 0x3FF0000000000000).bitcast("f64") - 1.0


if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")
    # dr.set_log_level(dr.LogLevel.Trace)

    m_v0 = mi.UInt32([0, 1, 2, 3])
    m_v1 = mi.UInt32([4, 5, 6, 7])

    m_v0, m_v1 = mi.sample_tea_32(m_v0, m_v1)

    print(f"{m_v0=}, {m_v1=}")

    pyjit.set_backend("optix")

    v0 = pyjit.u32([0, 1, 2, 3])
    v1 = pyjit.u32([4, 5, 6, 7])

    v0, v1 = sample_tea_32(v0, v1)

    print(f"{pyjit.u64(v0)=}")
    print(f"{v0=}, {v1=}")

    m_v0 = mi.UInt32([0, 1, 2, 3])
    m_v1 = mi.UInt32([4, 5, 6, 7])

    m = mi.sample_tea_64(m_v0, m_v1)

    print(f"{m=}")

    v0 = pyjit.u32([0, 1, 2, 3])
    v1 = pyjit.u32([4, 5, 6, 7])

    v = sample_tea_64(v0, v1)

    print(f"{v=}")

    rng = PCG32(10)

    print(f"own: {rng.next_u32()=}")
    print(f"onw: {rng.next_f32()=}")

    rng = mi.PCG32(10)

    print(f"mitsuba: {rng.next_uint32()=}")
    print(f"mitsuba: {rng.next_float32()=}")
    # print(f"{(rng.next_uint32() | 0x3F800000) >> 9}")
