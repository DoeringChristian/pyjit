use super::var::Var;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rjit::backend::CompileOptions;
use rjit::{Jit, Trace};

pub static IR: Lazy<Trace> = Lazy::new(|| Trace::default());

static JIT: Lazy<Mutex<Jit>> = Lazy::new(|| Mutex::new(Jit::default()));

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
pub fn set_backend(backend: &str) {
    IR.set_backend(backend)
}

#[pyfunction]
pub fn eval() {
    JIT.lock().eval(&mut IR.lock())
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
                Var::from_any_of(value, rjit::VarType::[<$ty:camel>])
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
