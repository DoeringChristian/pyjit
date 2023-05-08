from dataclasses import dataclass
from point import Point3f
from vector import Vector3f


@dataclass
class Ray3f:
    o: Point3f
    d: Vector3f
