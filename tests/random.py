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


if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")

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
