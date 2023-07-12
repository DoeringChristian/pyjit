use crate::funcs::{self, IR};
use anyhow::Result;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyList;
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
            VarType::Bool => funcs::bool(any, None),
            VarType::I8 => funcs::i8(any, None),
            VarType::U8 => funcs::u8(any, None),
            VarType::I16 => funcs::i16(any, None),
            VarType::U16 => funcs::u16(any, None),
            VarType::I32 => funcs::i32(any, None),
            VarType::U32 => funcs::u32(any, None),
            VarType::I64 => funcs::i64(any, None),
            VarType::U64 => funcs::u64(any, None),
            VarType::F16 => todo!(),
            VarType::F32 => funcs::f32(any, None),
            VarType::F64 => funcs::f64(any, None),
        }
    }
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
        Ok(Var(self.0.add(&other.0)?))
    }
    pub fn sub(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.sub(&other.0)?))
    }
    pub fn mul(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.mul(&other.0)?))
    }
    pub fn div(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.div(&other.0)?))
    }
    pub fn modulo(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.modulo(&other.0)?))
    }
    pub fn and(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.and(&other.0)?))
    }
    pub fn rcp(&self) -> PyResult<Self> {
        Ok(Var(self.0.rcp()?))
    }
    pub fn rsqrt(&self) -> PyResult<Self> {
        Ok(Var(self.0.rsqrt()?))
    }
    pub fn sin(&self) -> PyResult<Self> {
        Ok(Var(self.0.sin()?))
    }
    pub fn cos(&self) -> PyResult<Self> {
        Ok(Var(self.0.cos()?))
    }
    pub fn exp2(&self) -> PyResult<Self> {
        Ok(Var(self.0.exp2()?))
    }
    pub fn log2(&self) -> PyResult<Self> {
        Ok(Var(self.0.log2()?))
    }
    pub fn neg(&self) -> PyResult<Self> {
        Ok(Var(self.0.neg()?))
    }
    pub fn not(&self) -> PyResult<Self> {
        Ok(Var(self.0.not()?))
    }
    pub fn abs(&self) -> PyResult<Self> {
        Ok(Var(self.0.abs()?))
    }
    pub fn ceil(&self) -> PyResult<Self> {
        Ok(Var(self.0.floor()?))
    }
    pub fn trunc(&self) -> PyResult<Self> {
        Ok(Var(self.0.trunc()?))
    }
    pub fn popc(&self) -> PyResult<Self> {
        Ok(Var(self.0.popc()?))
    }
    pub fn clz(&self) -> PyResult<Self> {
        Ok(Var(self.0.clz()?))
    }
    pub fn ctz(&self) -> PyResult<Self> {
        Ok(Var(self.0.ctz()?))
    }
    pub fn min(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.min(&other.0)?))
    }
    pub fn max(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.max(&other.0)?))
    }
    pub fn eq(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.eq(&other.0)?))
    }
    pub fn neq(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.neq(&other.0)?))
    }
    pub fn lt(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.lt(&other.0)?))
    }
    pub fn le(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.le(&other.0)?))
    }
    pub fn gt(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.gt(&other.0)?))
    }
    pub fn ge(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.ge(&other.0)?))
    }

    pub fn or(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.or(&other.0)?))
    }
    pub fn xor(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.xor(&other.0)?))
    }
    pub fn shl(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.shl(&other.0)?))
    }
    pub fn shr(&self, other: &PyAny) -> PyResult<Self> {
        let other = Self::from_any_of(other, self.0.ty())?;
        Ok(Var(self.0.shr(&other.0)?))
    }

    pub fn fma(&self, d1: &PyAny, d2: &PyAny) -> PyResult<Self> {
        let d1 = Self::from_any_of(d1, self.0.ty())?;
        let d2 = Self::from_any_of(d2, self.0.ty())?;
        Ok(Var(self.0.fma(&d1.0, &d2.0)?))
    }
    pub fn select(&self, d1: &PyAny, d2: &PyAny) -> PyResult<Self> {
        let d1 = Self::from_any_of(d1, self.0.ty())?;
        let d2 = Self::from_any_of(d2, self.0.ty())?;
        Ok(Var(self.0.select(&d1.0, &d2.0)?))
    }

    pub fn __add__(&self, other: &PyAny) -> PyResult<Self> {
        self.add(other)
    }
    pub fn __sub__(&self, other: &PyAny) -> PyResult<Self> {
        self.sub(other)
    }
    pub fn __mul__(&self, other: &PyAny) -> PyResult<Self> {
        self.mul(other)
    }
    pub fn __div__(&self, other: &PyAny) -> PyResult<Self> {
        self.div(other)
    }
    pub fn __or__(&self, other: &PyAny) -> PyResult<Self> {
        self.or(other)
    }
    pub fn __and__(&self, other: &PyAny) -> PyResult<Self> {
        self.and(other)
    }
    pub fn __xor__(&self, other: &PyAny) -> PyResult<Self> {
        self.xor(other)
    }

    pub fn __iadd__(&mut self, other: &PyAny) -> PyResult<()> {
        *self = self.add(other)?;
        Ok(())
    }
    pub fn __isub__(&mut self, other: &PyAny) -> PyResult<()> {
        *self = self.sub(other)?;
        Ok(())
    }
    pub fn __imul__(&mut self, other: &PyAny) -> PyResult<()> {
        *self = self.mul(other)?;
        Ok(())
    }
    pub fn __idiv__(&mut self, other: &PyAny) -> PyResult<()> {
        *self = self.div(other)?;
        Ok(())
    }
    pub fn __ior__(&mut self, other: &PyAny) -> PyResult<()> {
        *self = self.or(other)?;
        Ok(())
    }
    pub fn __iand__(&mut self, other: &PyAny) -> PyResult<()> {
        *self = self.and(other)?;
        Ok(())
    }
    pub fn __ixor__(&mut self, other: &PyAny) -> PyResult<()> {
        *self = self.xor(other)?;
        Ok(())
    }

    pub fn bitcast(&self, ty: &str) -> PyResult<Self> {
        let ty = ty.to_lowercase();
        match ty.as_str() {
            "bool" => Ok(Self(self.0.bitcast(&VarType::Bool)?)),
            "u8" => Ok(Self(self.0.bitcast(&VarType::U8)?)),
            "i8" => Ok(Self(self.0.bitcast(&VarType::I8)?)),
            "i16" => Ok(Self(self.0.bitcast(&VarType::I16)?)),
            "u16" => Ok(Self(self.0.bitcast(&VarType::U16)?)),
            "i32" => Ok(Self(self.0.bitcast(&VarType::I32)?)),
            "u32" => Ok(Self(self.0.bitcast(&VarType::U32)?)),
            "i64" => Ok(Self(self.0.bitcast(&VarType::I64)?)),
            "u64" => Ok(Self(self.0.bitcast(&VarType::U64)?)),
            "f32" => Ok(Self(self.0.bitcast(&VarType::F32)?)),
            "f64" => Ok(Self(self.0.bitcast(&VarType::F64)?)),
            _ => Err(PyErr::new::<PyTypeError, _>(format!(
                "Type {ty} is not supported!"
            ))),
        }
    }

    pub fn to_texture(&self, shape: Vec<usize>, n_channels: usize) -> PyResult<Self> {
        Ok(Var(self.0.to_texture(&shape, n_channels)?))
    }

    pub fn tex_to_buffer(&self) -> PyResult<Self> {
        Ok(Var(self.0.tex_to_buffer()?))
    }

    pub fn tex_lookup(&self, pos: Vec<&PyAny>) -> PyResult<Vec<Self>> {
        let pos = pos
            .iter()
            .map(|p| Self::from_any_of(p, VarType::F32).unwrap().0)
            .collect::<Vec<_>>();
        let pos_refs = pos.iter().map(|p| p).collect::<Vec<_>>();
        let res = self.0.tex_lookup(pos_refs.as_slice())?;
        let res = res.into_iter().map(|r| Var(r)).collect::<_>();
        Ok(res)
    }

    pub fn compress(&self) -> Result<Self> {
        Ok(Var(self.0.compress()?))
    }

    pub fn scatter_reduce(&self, dst: &Self, idx: &PyAny, mask: Option<&PyAny>) -> Result<()> {
        let mask = mask.map(|m| Self::from_any_of(m, VarType::Bool).unwrap().0);
        self.0.scatter_reduce(
            &dst.0,
            &Self::from_any_of(idx, VarType::U32).unwrap().0,
            mask.as_ref(),
            ReduceOp::Add,
        )
    }
    pub fn scatter(&self, dst: &Self, idx: &PyAny, mask: Option<&PyAny>) -> Result<()> {
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
        Ok(Var(self.0.gather(&idx, mask.as_ref())?))
    }
    pub fn trace_ray(
        &self,
        payload: Vec<&PyAny>,
        o: Vec<&PyAny>,
        d: Vec<&PyAny>,
        tmin: &PyAny,
        tmax: &PyAny,
        t: &PyAny,
        vis_mask: Option<&PyAny>,
        flags: Option<&PyAny>,
        sbt_offset: Option<&PyAny>,
        sbt_stride: Option<&PyAny>,
        miss_sbt: Option<&PyAny>,
        mask: Option<&PyAny>,
    ) -> PyResult<Vec<Self>> {
        let o = [
            &funcs::f32(o[0], None)?.0,
            &funcs::f32(o[1], None)?.0,
            &funcs::f32(o[2], None)?.0,
        ];
        let d = [
            &funcs::f32(d[0], None)?.0,
            &funcs::f32(d[1], None)?.0,
            &funcs::f32(d[2], None)?.0,
        ];
        let vis_mask = vis_mask.map(|v| funcs::u32(v, None).unwrap().0);
        let flags = flags.map(|v| funcs::u32(v, None).unwrap().0);
        let sbt_offset = sbt_offset.map(|v| funcs::u32(v, None).unwrap().0);
        let sbt_stride = sbt_stride.map(|v| funcs::u32(v, None).unwrap().0);
        let miss_sbt = miss_sbt.map(|v| funcs::u32(v, None).unwrap().0);
        let mask = mask.map(|v| funcs::bool(v, None).unwrap().0);
        let payload = payload
            .into_iter()
            .map(|v| funcs::u32(v, None).unwrap().0)
            .collect::<Vec<_>>();
        let payload_ref = payload.iter().collect::<Vec<_>>();

        Ok(self
            .0
            .trace_ray(
                &payload_ref,
                o,
                d,
                &funcs::f32(tmin, None)?.0,
                &funcs::f32(tmax, None)?.0,
                &funcs::f32(t, None)?.0,
                vis_mask.as_ref(),
                flags.as_ref(),
                sbt_offset.as_ref(),
                sbt_stride.as_ref(),
                miss_sbt.as_ref(),
                mask.as_ref(),
            )?
            .into_iter()
            .map(|p| Var(p))
            .collect::<Vec<_>>())
    }
    pub fn __repr__(&self) -> Result<String> {
        self.schedule();
        funcs::eval();
        Ok(match self.0.ty() {
            VarType::Void => format!(""),
            VarType::Bool => format!("bool{:?}", self.0.to_host::<bool>()?.as_slice()),
            VarType::I8 => format!("i8{:?}", self.0.to_host::<i8>()?.as_slice()),
            VarType::U8 => format!("u8{:?}", self.0.to_host::<u8>()?.as_slice()),
            VarType::I16 => format!("i16{:?}", self.0.to_host::<i16>()?.as_slice()),
            VarType::U16 => format!("u16{:?}", self.0.to_host::<u16>()?.as_slice()),
            VarType::I32 => format!("i32{:?}", self.0.to_host::<i32>()?.as_slice()),
            VarType::U32 => format!("u32{:?}", self.0.to_host::<u32>()?.as_slice()),
            VarType::I64 => format!("i64{:?}", self.0.to_host::<i64>()?.as_slice()),
            VarType::U64 => format!("u64{:?}", self.0.to_host::<u64>()?.as_slice()),
            VarType::F16 => todo!(),
            VarType::F32 => format!("f32{:?}", self.0.to_host::<f32>()?.as_slice()),
            VarType::F64 => format!("f64{:?}", self.0.to_host::<f64>()?.as_slice()),
        })
    }
    pub fn to_list<'a>(&self, py: Python<'a>) -> Result<&'a PyList> {
        Ok(match self.0.ty() {
            VarType::Void => todo!(),
            VarType::Bool => PyList::new(py, self.0.to_host::<bool>()?),
            VarType::I8 => PyList::new(py, self.0.to_host::<i8>()?),
            VarType::U8 => PyList::new(py, self.0.to_host::<u8>()?),
            VarType::I16 => PyList::new(py, self.0.to_host::<i16>()?),
            VarType::U16 => PyList::new(py, self.0.to_host::<u16>()?),
            VarType::I32 => PyList::new(py, self.0.to_host::<i32>()?),
            VarType::U32 => PyList::new(py, self.0.to_host::<u32>()?),
            VarType::I64 => PyList::new(py, self.0.to_host::<i64>()?),
            VarType::U64 => PyList::new(py, self.0.to_host::<u64>()?),
            VarType::F16 => todo!(),
            VarType::F32 => PyList::new(py, self.0.to_host::<f32>()?),
            VarType::F64 => PyList::new(py, self.0.to_host::<f64>()?),
        })
    }
    pub fn to_numpy<'a>(&self, py: Python<'a>) -> Result<&'a PyAny> {
        Ok(match self.0.ty() {
            VarType::Void => todo!(),
            VarType::Bool => numpy::PyArray1::<bool>::from_vec(py, self.0.to_host::<bool>()?),
            VarType::I8 => numpy::PyArray1::<i8>::from_vec(py, self.0.to_host::<i8>()?),
            VarType::U8 => numpy::PyArray1::<u8>::from_vec(py, self.0.to_host::<u8>()?),
            VarType::I16 => numpy::PyArray1::<i16>::from_vec(py, self.0.to_host::<i16>()?),
            VarType::U16 => numpy::PyArray1::<u16>::from_vec(py, self.0.to_host::<u16>()?),
            VarType::I32 => numpy::PyArray1::<i32>::from_vec(py, self.0.to_host::<i32>()?),
            VarType::U32 => numpy::PyArray1::<u32>::from_vec(py, self.0.to_host::<u32>()?),
            VarType::I64 => numpy::PyArray1::<i64>::from_vec(py, self.0.to_host::<i64>()?),
            VarType::U64 => numpy::PyArray1::<u64>::from_vec(py, self.0.to_host::<u64>()?),
            VarType::F16 => todo!(),
            VarType::F32 => numpy::PyArray1::<f32>::from_vec(py, self.0.to_host::<f32>()?),
            VarType::F64 => numpy::PyArray1::<f64>::from_vec(py, self.0.to_host::<f64>()?),
        })
    }
}
