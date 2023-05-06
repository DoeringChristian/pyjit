import pyjit

pyjit.set_backend("optix")

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

pyjit.set_compile_options(5)
pyjit.set_miss("__miss__ms", miss_and_closesthit_ptx)
pyjit.push_hit("__closesthit__ch", miss_and_closesthit_ptx)

indices = pyjit.u32([0, 1, 2])
vertices = pyjit.f32([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])

desc = pyjit.AccelDesc()
t0 = desc.add_triangles(vertices, indices)
desc.add_instance(t0, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])

accel: pyjit.Var = pyjit.accel(desc)

payload: list[pyjit.Var] = accel.trace_ray(
    5,
    [[0.6, 0.6], 0.6, 0.0],
    [0.0, 0.0, [1.0, 1.0]],
    0.001,
    1000.0,
    0.0,
)

valid: pyjit.Var = pyjit.bool(payload[0])
valid.schedule()

primitive_idx = payload[1]
instance_id = payload[2]
primitive_idx.schedule()
instance_id.schedule()

u = payload[3].bitcast("f32")
v = payload[4].bitcast("f32")
u.schedule()
v.schedule()

pyjit.eval()

print(f"{valid=}")
print(f"{primitive_idx=}")
print(f"{instance_id=}")
print(f"{u=}")
print(f"{v=}")
