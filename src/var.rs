use crate::funcs::IR;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use rjit::{ReduceOp, VarType};

#[pyclass]
#[derive(Clone)]
pub struct Var(pub rjit::VarRef);

macro_rules! match_return {
    ($any:ident,$ty:ident) => {
        paste::paste! {
            if let Ok(val) = $any.extract::<Self>(){
                if val.0.ty() == VarType::[<$ty:camel>] {
                    return Ok(val);
                } else {
                    return Ok(Var(val.0.cast(&VarType::[<$ty:camel>])));
                }
            }
            if let Ok(val) = $any.extract::<$ty>() {
                return Ok(Self(IR.[<literal_$ty>](val)));
            }
            if let Ok(val) = $any.extract::<Vec<$ty>>() {
                return Ok(Self(IR.[<buffer_$ty>](&val)));
            }
        }
    };
}

impl Var {
    pub fn from_any_of(any: &PyAny, ty: VarType) -> PyResult<Self> {
        match ty {
            VarType::Void => todo!(),
            VarType::Bool => {
                match_return!(any, bool);
            }
            VarType::I8 => {
                match_return!(any, i8);
            }
            VarType::U8 => {
                match_return!(any, u8);
            }
            VarType::I16 => {
                match_return!(any, i16);
            }
            VarType::U16 => {
                match_return!(any, u16);
            }
            VarType::I32 => {
                match_return!(any, i32);
            }
            VarType::U32 => {
                match_return!(any, u32);
            }
            VarType::I64 => {
                match_return!(any, i64);
            }
            VarType::U64 => {
                match_return!(any, u64);
            }
            VarType::F16 => todo!(),
            VarType::F32 => {
                match_return!(any, f32);
            }
            VarType::F64 => {
                match_return!(any, f64);
            }
        }
        Err(PyErr::new::<PyTypeError, _>(
            format!(
                "Could not cast python object of type {} to type {:?}",
                any.get_type().str().unwrap(),
                ty
            ), // "Could not cast python type to jit type!",
        ))
    }
}

macro_rules! bop {
    ($op:ident) => {
        pub fn $op(&self, other: &PyAny) -> PyResult<Self> {
            let other = Self::from_any_of(other, self.0.ty())?;
            Ok(Var(self.0.$op(&other.0)))
        }
    };
}

macro_rules! uop {
    ($op:ident) => {
        pub fn $op(&self) -> PyResult<Self> {
            Ok(Var(self.0.$op()))
        }
    };
}

#[pymethods]
impl Var {
    pub fn ty(&self) -> String {
        format!("{:?}", self.0.ty())
    }
    pub fn size(&self) -> usize {
        self.0.size()
    }
    pub fn schedule(&self) {
        self.0.schedule()
    }

    pub fn add(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.add(&other.0)))
    }
    pub fn sub(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.sub(&other.0)))
    }
    pub fn mul(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.mul(&other.0)))
    }
    pub fn div(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.div(&other.0)))
    }
    pub fn and(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.and(&other.0)))
    }
    pub fn rcp(&self) -> PyResult<Self> {
        Ok(Var(self.0.rcp()))
    }
    pub fn rsqrt(&self) -> PyResult<Self> {
        Ok(Var(self.0.rsqrt()))
    }
    pub fn sin(&self) -> PyResult<Self> {
        Ok(Var(self.0.sin()))
    }
    pub fn cos(&self) -> PyResult<Self> {
        Ok(Var(self.0.cos()))
    }
    pub fn exp2(&self) -> PyResult<Self> {
        Ok(Var(self.0.exp2()))
    }
    pub fn log2(&self) -> PyResult<Self> {
        Ok(Var(self.0.log2()))
    }

    pub fn to_texture(&self, shape: Vec<usize>, n_channels: usize) -> PyResult<Self> {
        Ok(Var(self.0.to_texture(&shape, n_channels)))
    }

    pub fn tex_to_buffer(&self) -> PyResult<Self> {
        Ok(Var(self.0.tex_to_buffer()))
    }

    pub fn tex_lookup(&self, pos: Vec<&PyAny>) -> PyResult<Vec<Self>> {
        let pos = pos
            .iter()
            .map(|p| Self::from_any_of(p, VarType::F32).unwrap().0)
            .collect::<Vec<_>>();
        let pos_refs = pos.iter().map(|p| p).collect::<Vec<_>>();
        let res = self.0.tex_lookup(pos_refs.as_slice());
        let res = res.into_iter().map(|r| Var(r)).collect::<Vec<_>>();
        Ok(res)
    }

    pub fn scatter_reduce(&self, dst: &Self, idx: &PyAny, mask: Option<&PyAny>) {
        let mask = mask.map(|m| Self::from_any_of(m, VarType::Bool).unwrap().0);
        self.0.scatter_reduce(
            &dst.0,
            &Self::from_any_of(idx, VarType::U32).unwrap().0,
            mask.as_ref(),
            ReduceOp::Add,
        )
    }
    pub fn scatter(&self, dst: &Self, idx: &PyAny, mask: Option<&PyAny>) {
        let mask = mask.map(|m| Self::from_any_of(m, VarType::Bool).unwrap().0);
        self.0.scatter(
            &dst.0,
            &Self::from_any_of(idx, VarType::U32).unwrap().0,
            mask.as_ref(),
        )
    }
    pub fn gather(&self, idx: &PyAny, mask: Option<&PyAny>) -> PyResult<Self> {
        let mask = mask.map(|m| Self::from_any_of(m, VarType::Bool).unwrap().0);
        let idx = Self::from_any_of(idx, VarType::U32)?.0;
        Ok(Var(self.0.gather(&idx, mask.as_ref())))
    }
}
