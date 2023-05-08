import pyjit
from dataclasses import dataclass


@dataclass
class Point2f:
    x: pyjit.Var
    y: pyjit.Var

    def __init__(self, x, y):
        self.x = pyjit.f32(x)
        self.y = pyjit.f32(y)


@dataclass
class Point3f:
    x: pyjit.Var
    y: pyjit.Var
    z: pyjit.Var

    def __init__(self, x, y, z):
        self.x = pyjit.f32(x)
        self.y = pyjit.f32(y)
        self.z = pyjit.f32(z)
