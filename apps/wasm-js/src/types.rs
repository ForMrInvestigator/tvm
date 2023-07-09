use wasm_bindgen::prelude::*;
use ndarray::{Array, ArrayD};



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
        Tensor {
            shape,
            data,
        }
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
