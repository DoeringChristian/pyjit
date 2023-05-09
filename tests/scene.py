import pyjit
from typing import Any
from sensor import Sensor, sensors, new_sensor
from shape import Shape, Instance, shapes, new_shape
from integrator import Integrator, integrators, new_integrator
from ray import Ray3f
from point import Point3f, Point2f
from dataclasses import dataclass


miss_and_closesthit_ptx = """
.version 8.0
.target sm_86
.address_size 64

.entry __miss__ms() {
    .reg .b32 %r<6>;
    mov.b32 %r0, 0;
    mov.b32 %r1, 0;

    call _optix_set_payload, (%r0, %r1);
    ret;
}

.entry __closesthit__ch() {
    .reg .b32 %i<5>;
    .reg .b32 %v<5>;
    mov.b32 %i0, 0;
    mov.b32 %i1, 1;
    mov.b32 %i2, 2;
    mov.b32 %i3, 3;
    mov.b32 %i4, 4;

        mov.b32 %v0, 1;
    call _optix_set_payload, (%i0, %v0);

        call (%v1), _optix_read_primitive_idx, ();
    call _optix_set_payload, (%i1, %v1);

        call (%v2), _optix_read_instance_id, ();
    call _optix_set_payload, (%i2, %v2);

    .reg .f32 %f<2>;
        call (%f0, %f1), _optix_get_triangle_barycentrics, ();
        mov.b32 %v3, %f0;
        mov.b32 %v4, %f1;
    call _optix_set_payload, (%i3, %v3);
    call _optix_set_payload, (%i4, %v4);

    ret;
}
"""


@dataclass
class PreliminaryInteraction:
    valid: pyjit.Var
    primitive_idx: pyjit.Var
    instance_id: pyjit.Var
    uv: Point2f


class Scene:
    accel: pyjit.Var
    sensors: list[Sensor] = []
    shapes: list[Shape] = []
    instances: list[Instance] = []
    integrators: list[Integrator] = []

    def __init__(self, desc: dict[str, Any]):
        pyjit.set_compile_options(5)
        pyjit.set_miss("__miss__ms", miss_and_closesthit_ptx)
        pyjit.push_hit("__closesthit__ch", miss_and_closesthit_ptx)

        self.acceldesc = pyjit.AccelDesc()
        geometries = {}
        for k, v in desc.items():
            if v["type"] in shapes:
                shape = new_shape(v.copy())
                geometries[k] = shape.add_to_accel(self.acceldesc)
                self.shapes.append(shape)
        for k, v in desc.items():
            if v["type"] == "instance":
                self.instances.append(Instance(geometries, self.acceldesc, v.copy()))

        for k, v in desc.items():
            if v["type"] in sensors:
                self.sensors.append(new_sensor(v.copy()))
            if v["type"] in integrators:
                self.integrators.append(new_integrator(v.copy()))

        self.accel: pyjit.Var = pyjit.accel(self.acceldesc)

    def intersect_preliminary(self, ray: Ray3f):
        payload = self.accel.trace_ray(
            [0, 0, 0, 0, 0],
            [ray.o.x, ray.o.y, ray.o.z],
            [ray.d.x, ray.d.y, ray.d.z],
            0.0001,
            1000.0,
            0.0,
        )
        return PreliminaryInteraction(
            pyjit.bool(payload[0]),
            payload[1],
            payload[2],
            Point2f(payload[3].bitcast("f32"), payload[4].bitcast("f32")),
        )


if __name__ == "__main__":
    pyjit.set_backend("optix")
    scene = Scene(
        {
            "m0": {
                "type": "mesh",
                "indices": [0, 1, 2],
                "vertices": [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            },
            "i0": {
                "type": "instance",
                "ref": "m0",
            },
            "integrator": {
                "type": "path",
            },
            "sensor": {
                "type": "orthogonal",
            },
        }
    )

    result = scene.integrators[0].render(scene)

    import matplotlib.pyplot as plt

    plt.imshow(result.to_numpy())
    plt.show()

    # print(f"{scene.sensors=}")
