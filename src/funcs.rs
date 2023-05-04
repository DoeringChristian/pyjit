use super::var::Var;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rjit::{Jit, Trace};

pub static IR: Lazy<Trace> = Lazy::new(|| Trace::default());

static JIT: Lazy<Mutex<Jit>> = Lazy::new(|| Mutex::new(Jit::default()));

#[pyfunction]
pub fn eval() {
    JIT.lock().eval(&mut IR.lock())
}

#[pyfunction]
pub fn set_backend(backend: &str) {
    IR.set_backend(backend)
}

#[pyfunction]
pub fn index(num: usize) -> Var {
    Var(IR.index(num))
}

#[pyfunction]
pub fn texture(shape: Vec<usize>, n_channels: usize) -> Var {
    Var(IR.texture(&shape, n_channels))
}

#[pyfunction]
pub fn accel(vertices: &Var, indices: &Var) -> Var {
    Var(IR.accel(&vertices.0, &indices.0))
}

macro_rules! initializer {
    ($ty:ident) => {
        paste::paste! {
            #[pyfunction]
            // #[pyo3(signature = (*args))]
            pub fn $ty(value: &PyAny) -> PyResult<Var> {
                if let Ok(lit) = value.extract::<$ty>() {
                    Ok(Var(IR.[<literal_$ty>](lit)))
                } else {
                    Ok(Var(IR.[<buffer_$ty>](value.extract::<Vec<$ty>>()?.as_slice())))
                }
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
