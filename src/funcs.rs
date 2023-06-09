use super::var::Var;
use anyhow::Result;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rjit::backend::CompileOptions;
use rjit::{Jit, Trace};

pub static IR: Lazy<Trace> = Lazy::new(|| Trace::default());

#[pyfunction]
pub fn set_miss(enty_point: &str, source: &str) {
    IR.backend().set_miss_from_str(enty_point, source);
}

#[pyfunction]
pub fn push_hit(enty_point: &str, source: &str) {
    IR.backend().push_hit_from_str(enty_point, source);
}

#[pyfunction]
pub fn set_compile_options(num_payload_values: u32) {
    IR.backend().set_compile_options(&CompileOptions {
        num_payload_values: num_payload_values as _,
    });
}

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
}

#[pyclass]
#[derive(Default)]
pub struct AccelDesc {
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
    pub fn add_instance(&mut self, geometry: usize, transform: [f32; 12]) {
        self.instances.push(InstanceDesc {
            geometry,
            transform,
        })
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
        })
        .collect::<Vec<_>>();
    let desc = rjit::AccelDesc {
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
