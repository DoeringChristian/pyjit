use super::var::Var;
use anyhow::Result;
use once_cell::sync::Lazy;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use rjit::Trace;

pub static IR: Lazy<Trace> = Lazy::new(|| Trace::default());

#[pyfunction]
pub fn set_backend(backend: &str) -> Result<()> {
    IR.set_backend(&[backend])
}

#[pyfunction]
pub fn eval() {
    IR.eval();
}

#[pyfunction]
pub fn index(num: usize) -> Var {
    Var(IR.index(num))
}

#[pyfunction]
pub fn texture(shape: Vec<usize>, n_channels: usize) -> Result<Var> {
    Ok(Var(IR.texture(&shape, n_channels)?))
}

enum GeometryDesc {
    Triangles { vertices: Var, indices: Var },
}
struct InstanceDesc {
    pub geometry: usize,
    pub transform: [f32; 12],
    pub hit_group: u32,
}

pub struct ModuleDesc {
    pub asm: String,
    pub entry_point: String,
}

pub struct HitGroupDesc {
    pub closest_hit: ModuleDesc,
    pub any_hit: Option<ModuleDesc>,
    pub intersection: Option<ModuleDesc>,
}
pub struct MissGroupDesc {
    pub miss: ModuleDesc,
}

#[pyclass]
#[derive(Default)]
pub struct AccelDesc {
    hit_groups: Vec<HitGroupDesc>,
    miss_groups: Vec<MissGroupDesc>,
    geometries: Vec<GeometryDesc>,
    instances: Vec<InstanceDesc>,
}

#[pymethods]
impl AccelDesc {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_triangles(&mut self, vertices: &PyAny, indices: &PyAny) -> PyResult<usize> {
        let vertices = f32(vertices, None)?;
        let indices = u32(indices, None)?;
        let id = self.geometries.len();
        self.geometries.push(GeometryDesc::Triangles {
            vertices: vertices.clone(),
            indices: indices.clone(),
        });
        Ok(id)
    }
    pub fn add_instance(&mut self, geometry: usize, transform: [f32; 12], hit_group: u32) {
        self.instances.push(InstanceDesc {
            geometry,
            transform,
            hit_group,
        })
    }
    pub fn add_hit_group(
        &mut self,
        closest_hit_entry_point: &str,
        closest_hit_asm: &str,
        any_hit_entry_point: Option<&str>,
        any_hit_asm: Option<&str>,
        intersection_entry_point: Option<&str>,
        intersection_asm: Option<&str>,
    ) -> u32 {
        let any_hit = any_hit_asm.map(|ah| ModuleDesc {
            asm: ah.into(),
            entry_point: any_hit_entry_point.unwrap().into(),
        });
        let intersection = intersection_asm.map(|int| ModuleDesc {
            asm: int.into(),
            entry_point: intersection_entry_point.unwrap().into(),
        });

        let idx = self.hit_groups.len();
        self.hit_groups.push(HitGroupDesc {
            closest_hit: ModuleDesc {
                asm: closest_hit_asm.into(),
                entry_point: closest_hit_entry_point.into(),
            },
            any_hit,
            intersection,
        });
        idx as _
    }
    pub fn add_miss_group(&mut self, entry_point: &str, asm: &str) -> u32 {
        let idx = self.miss_groups.len();
        self.miss_groups.push(MissGroupDesc {
            miss: ModuleDesc {
                asm: asm.into(),
                entry_point: entry_point.into(),
            },
        });
        idx as _
    }
}

#[pyfunction]
pub fn accel(desc: &AccelDesc) -> Result<Var> {
    let geometries = desc
        .geometries
        .iter()
        .map(|g| match g {
            GeometryDesc::Triangles { vertices, indices } => rjit::GeometryDesc::Triangles {
                vertices: &vertices.0,
                indices: &indices.0,
            },
        })
        .collect::<Vec<_>>();
    let instances = desc
        .instances
        .iter()
        .map(|i| rjit::InstanceDesc {
            geometry: i.geometry,
            transform: i.transform,
            hit_group: i.hit_group,
        })
        .collect::<Vec<_>>();

    let hit_groups = desc
        .hit_groups
        .iter()
        .map(|hg| rjit::HitGroupDesc {
            closest_hit: rjit::ModuleDesc {
                asm: &hg.closest_hit.asm,
                entry_point: &hg.closest_hit.entry_point,
            },
            any_hit: hg.any_hit.as_ref().map(|ah| rjit::ModuleDesc {
                asm: &ah.asm,
                entry_point: &ah.entry_point,
            }),
            intersection: hg.intersection.as_ref().map(|int| rjit::ModuleDesc {
                asm: &int.asm,
                entry_point: &int.entry_point,
            }),
        })
        .collect::<Vec<_>>();

    let miss_groups = desc
        .miss_groups
        .iter()
        .map(|mg| rjit::MissGroupDesc {
            miss: rjit::ModuleDesc {
                asm: &mg.miss.asm,
                entry_point: &mg.miss.entry_point,
            },
        })
        .collect::<Vec<_>>();
    let sbt = rjit::SBTDesc {
        hit_groups: &hit_groups,
        miss_groups: &miss_groups,
    };
    let desc = rjit::AccelDesc {
        sbt,
        geometries: &geometries,
        instances: &instances,
    };
    Ok(Var(IR.accel(desc)?))
}

macro_rules! initializer {
    ($ty:ident) => {
        paste::paste! {
            #[pyfunction]
            pub fn $ty(value: &PyAny, num: Option<usize>) -> PyResult<Var> {
                if let Ok(val) = value.extract::<Var>(){
                    if val.0.ty() == rjit::VarType::[<$ty:camel>] {
                        return Ok(val);
                    } else {
                        return Ok(Var(val.0.cast(&rjit::VarType::[<$ty:camel>])?));
                    }
                }
                if let Ok(val) = value.extract::<$ty>() {
                    return Ok(Var(IR.sized_literal::<$ty>(val, num.unwrap_or(1))?));
                }
                if let Ok(val) = value.extract::<Vec<$ty>>() {
                    return Ok(Var(IR.array(&val)?));
                }
                if let Ok(val) = value.extract::<numpy::PyReadonlyArray1<$ty>>() {
                    return Ok(Var(IR.array(&val.to_vec()?)?));
                }

                Err(PyErr::new::<PyTypeError, _>(
                    format!(
                        "Could not cast python object of type {} to type {:?}",
                        value.get_type().str().unwrap(),
                        &rjit::VarType::[<$ty:camel>]
                    ), // "Could not cast python type to jit type!",
                ))
            }
        }
    };
}

initializer!(bool);
initializer!(i8);
initializer!(u8);
initializer!(i16);
initializer!(u16);
initializer!(i32);
initializer!(u32);
initializer!(i64);
initializer!(u64);
initializer!(f32);
initializer!(f64);
