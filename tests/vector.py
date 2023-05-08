import pyjit
from dataclasses import dataclass


@dataclass
class Vector2f:
    x: pyjit.Var
    y: pyjit.Var

    def __init__(self, x, y):
        self.x = pyjit.f32(x)
        self.y = pyjit.f32(y)
        self.y = pyjit.f32(y)


@dataclass
class Vector3f:
    x: pyjit.Var
    y: pyjit.Var
    z: pyjit.Var

    def __init__(self, x, y, z):
        self.x = pyjit.f32(x)
        self.y = pyjit.f32(y)
        self.z = pyjit.f32(z)


@dataclass
class Vector4f:
    x: pyjit.Var
    y: pyjit.Var
    z: pyjit.Var
    w: pyjit.Var

    def __init__(self, x, y, z, w):
        self.x = pyjit.f32(x)
        self.y = pyjit.f32(y)
        self.z = pyjit.f32(z)
        self.w = pyjit.f32(w)
