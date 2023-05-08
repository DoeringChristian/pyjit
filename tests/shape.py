import pyjit
from transform import Matrix4f


shapes = {}


def register_shape(name: str, init):
    shapes[name] = init


def new_shape(desc: dict) -> "Shape":
    return shapes[desc.pop("type")](desc.copy())


class Shape:
    def __init__(self, desc: dict):
        ...

    def add_to_accel(self, acceldesc: pyjit.AccelDesc) -> int:
        ...


class Mesh(Shape):
    def __init__(self, desc: dict):
        self.vertices = pyjit.f32(desc.pop("vertices"))
        self.indices = pyjit.u32(desc.pop("indices"))

    def add_to_accel(self, acceldesc: pyjit.AccelDesc) -> int:
        return acceldesc.add_triangles(self.vertices, self.indices)


register_shape("mesh", lambda desc: Mesh(desc))


class Instance:
    def __init__(self, geometries: dict, acceldesc: pyjit.AccelDesc, desc: dict):
        self.ref = desc.pop("ref")
        self.to_world = Matrix4f(desc.pop("to_world", Matrix4f.ident()))
        self.geometry = geometries[self.ref]
        acceldesc.add_instance(self.geometry, self.to_world.to_4x3())


if __name__ == "__main__":
    pyjit.set_backend("optix")
    mesh = Mesh(
        {
            "indices": [0, 1, 2],
            "vertices": [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
