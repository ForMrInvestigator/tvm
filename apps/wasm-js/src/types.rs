extern crate web_sys;

use ndarray::{Array, ArrayD};
use wasm_bindgen::prelude::*;
use web_sys::console;

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub(crate) shape: Vec<usize>,
    pub(crate) data: Vec<f32>,
}

#[wasm_bindgen]
#[allow(dead_code)]
impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Tensor { shape, data }
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    pub fn data(&self) -> Vec<f32> {
        self.data.clone()
    }
}

impl From<ArrayD<f32>> for Tensor {
    fn from(value: ArrayD<f32>) -> Self {
        Tensor::new(value.shape().into(), value.into_raw_vec())
    }
}

impl Into<ArrayD<f32>> for Tensor {
    fn into(self) -> ArrayD<f32> {
        Array::from_shape_vec(self.shape(), self.data().into()).unwrap()
    }
}

pub struct Timer<'a> {
    name: &'a str,
}

impl<'a> Timer<'a> {
    pub fn new(name: &'a str) -> Timer<'a> {
        console::time_with_label(name);
        Timer { name }
    }
}

impl<'a> Drop for Timer<'a> {
    fn drop(&mut self) {
        console::time_end_with_label(self.name);
    }
}
