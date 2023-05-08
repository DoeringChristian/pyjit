import pyjit
from typing import Any
from sensor import Sensor, sensors

shapes = {}


def register_shape(name: str, init):
    shapes[name] = init


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


class Scene:
    accel: pyjit.Var
    sensors: list[Sensor] = []

    def __init__(self, desc: dict[str, Any]):
        pyjit.set_compile_options(5)
        pyjit.set_miss("__miss__ms", miss_and_closesthit_ptx)
        pyjit.push_hit("__closesthit__ch", miss_and_closesthit_ptx)

        adesc = pyjit.AccelDesc()
        k2g = {}
        for k, v in desc.items():
            if v["type"] == "mesh":
                k2g[k] = adesc.add_triangles(
                    vertices=v["vertices"],
                    indices=v["indices"],
                )

        for k, v in desc.items():
            if v["type"] == "instance":
                adesc.add_instance(geometry=k2g[v["ref"]], transform=v["to_world"])

        for k, v in desc.items():
            if sensors[v["type"]] is not None:
                self.sensors.append(sensors[v["type"]](v))

        self.accel = pyjit.accel(adesc)


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
                "to_world": [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
            },
        }
    )
