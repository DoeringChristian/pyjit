use pyo3::prelude::*;

use self::funcs::*;
use self::var::*;

mod funcs;
mod var;

// /// Formats the sum of two numbers as string.
// #[pyfunction]
// fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
//     Ok((a + b).to_string())
// }

/// A Python module implemented in Rust.
#[pymodule]
fn pyjit(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<Var>()?;
    m.add_class::<AccelDesc>()?;

    m.add_function(wrap_pyfunction!(funcs::bool, m)?)?;
    m.add_function(wrap_pyfunction!(funcs::i8, m)?)?;
    m.add_function(wrap_pyfunction!(funcs::u8, m)?)?;
    m.add_function(wrap_pyfunction!(funcs::i16, m)?)?;
    m.add_function(wrap_pyfunction!(funcs::u16, m)?)?;
    m.add_function(wrap_pyfunction!(funcs::i32, m)?)?;
    m.add_function(wrap_pyfunction!(funcs::u32, m)?)?;
    m.add_function(wrap_pyfunction!(funcs::i64, m)?)?;
    m.add_function(wrap_pyfunction!(funcs::u64, m)?)?;
    m.add_function(wrap_pyfunction!(funcs::f32, m)?)?;
    m.add_function(wrap_pyfunction!(funcs::f64, m)?)?;
    m.add_function(wrap_pyfunction!(funcs::i32, m)?)?;

    m.add_function(wrap_pyfunction!(funcs::index, m)?)?;
    m.add_function(wrap_pyfunction!(funcs::texture, m)?)?;
    m.add_function(wrap_pyfunction!(funcs::accel, m)?)?;

    m.add_function(wrap_pyfunction!(funcs::set_backend, m)?)?;
    m.add_function(wrap_pyfunction!(funcs::set_miss, m)?)?;
    m.add_function(wrap_pyfunction!(funcs::push_hit, m)?)?;
    m.add_function(wrap_pyfunction!(funcs::set_compile_options, m)?)?;

    m.add_function(wrap_pyfunction!(funcs::eval, m)?)?;
    Ok(())
}
