from dataclasses import dataclass
from typing import Any
import copy
import numpy as np
import pyjit
from point import Point3f
from vector import Vector4f


class Matrix4f:
    def __init__(self, arg):
        if isinstance(arg, Matrix4f):
            self.cols = copy.deepcopy(arg.cols)
        elif isinstance(arg, list):
            self.cols = [[e for e in col] for col in arg]

    def __mul__(self, other):
        if isinstance(other, Matrix4f):
            ...
        elif isinstance(other, Point3f):
            self.__mul__(Vector4f(other.x, other.y, other.z, 1.0))
        elif isinstance(other, Vector4f):
            x = (
                other.x * self.cols[0][0]
                + other.y * self.cols[1][0]
                + other.z * self.cols[2][0]
                + other.w * self.cols[3][0]
            )

            y = (
                other.x * self.cols[0][1]
                + other.y * self.cols[1][1]
                + other.z * self.cols[2][1]
                + other.w * self.cols[3][1]
            )
            z = (
                other.x * self.cols[0][2]
                + other.y * self.cols[1][2]
                + other.z * self.cols[2][2]
                + other.w * self.cols[3][2]
            )
            w = (
                other.x * self.cols[0][3]
                + other.y * self.cols[1][3]
                + other.z * self.cols[2][3]
                + other.w * self.cols[3][3]
            )

            return Vector4f(x, y, z, w)

    @classmethod
    def ident(cls) -> "Matrix4f":
        return cls(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def to_3x4(self) -> list[float]:
        return [
            self.cols[0][0],
            self.cols[1][0],
            self.cols[2][0],
            self.cols[3][0],
            self.cols[0][1],
            self.cols[1][1],
            self.cols[2][1],
            self.cols[3][1],
            self.cols[0][2],
            self.cols[1][2],
            self.cols[2][2],
            self.cols[3][2],
        ]


class Transform4f:
    def __init__(self, arg):
        if isinstance(arg, list[list]):
            self.matrix = Matrix4f(self)
