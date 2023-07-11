#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate serde_derive;
extern crate console_error_panic_hook;
extern crate wee_alloc;

mod types;

use std::cmp::min;
use std::{collections::HashMap, convert::TryFrom};

use std::convert::TryInto;
use std::sync::Once;

use image::imageops::FilterType;
use image::{DynamicImage, RgbaImage};

use ndarray::prelude::*;
use ndarray::{array, s, Array};

use log::Level;
use log::{debug, info};
use tvm_graph_rt::{Graph, GraphExecutor, SystemLibModule, Tensor as TVMTensor};
use wasm_bindgen::prelude::*;

use crate::types::Timer;
use types::Tensor;

extern "C" {
    fn __wasm_call_ctors();
}

lazy_static! {
    static ref SYSLIB: SystemLibModule = SystemLibModule::default();
}

static mut EXEC: Option<GraphExecutor> = None;

static ONCE: Once = Once::new();

#[no_mangle]
#[wasm_bindgen]
pub fn pipeline(graph: &str, param: &[u8], tensor: Tensor) -> Tensor {
    let _global = Timer::new("WASM::pipeline::global");
    ONCE.call_once(|| unsafe {
        let _init = Timer::new("WASM::pipeline::init");
        // This is necessary to invoke TVMBackendRegisterSystemLibSymbol
        // API calls.
        __wasm_call_ctors();

        let graph = Graph::try_from(graph).unwrap();

        let params = tvm_graph_rt::load_param_dict(param)
            .unwrap()
            .into_iter()
            .map(|(k, v)| (k, v.to_owned()))
            .collect::<HashMap<String, TVMTensor<'static>>>();
        let mut exector = GraphExecutor::new(graph, &*SYSLIB).unwrap();
        exector.load_params(params);
        EXEC = Some(exector);
        info!("exec init");
    });

    // tensor to ndarray to image;
    let input: ArrayD<f32> = tensor.into();
    debug!("input shape: {:#?}", input.shape());
    debug!("input ndim: {:#?}", input.ndim());
    debug!("input data 2x2x4: {:#?}", input.slice(s![0..2, 0..2, ..]));

    let mut image: DynamicImage = RgbaImage::from_vec(
        input.shape()[0] as u32,
        input.shape()[1] as u32,
        input.mapv(|x| x as u8).into_raw_vec(),
    )
    .unwrap()
    .into();

    // center crop and scale and to rgb
    let scale = 224;
    let crop_size = min(image.height(), image.width());
    let rgb = image
        .crop(
            (image.width() - crop_size) / 2,
            (image.height() - crop_size) / 2,
            crop_size,
            crop_size,
        )
        .resize(scale, scale, FilterType::Nearest)
        .to_rgb8();

    // to ndarray again
    let mut input = Array::from_shape_vec([scale as usize, scale as usize, 3], rgb.to_vec())
        .unwrap()
        .mapv(|x| x as f32);

    // normalization
    // input * radio - mean * reverse_std
    let mean = [0.485, 0.456, 0.406];
    let reverse_div = [4.367, 4.464, 4.444];
    let ratio = 1.0 / 255.0;
    let normalized_div = array![
        ratio * reverse_div[0],
        ratio * reverse_div[1],
        ratio * reverse_div[2]
    ];
    let normalized_mean = array![
        mean[0] * reverse_div[0],
        mean[1] * reverse_div[1],
        mean[2] * reverse_div[2]
    ];
    input = input * normalized_div - normalized_mean;

    //HWC to CHW
    input = input.permuted_axes([2, 0, 1]);

    //clone items to make the memory layout contiguous;
    let contiguous_input =
        Array::from_shape_vec(input.raw_dim(), input.iter().cloned().collect()).unwrap();

    unsafe {
        // set input tensor and run the graph get output;
        let mut executor = EXEC.take().unwrap();
        executor.set_input("input", contiguous_input.into());
        executor.run();
        let output: ArrayD<f32> = executor
            .get_output(0)
            .unwrap()
            .to_owned()
            .try_into()
            .unwrap();
        EXEC.replace(executor);

        // softmax;
        let exps = output.map(|x| x.exp());
        let sum_of_exps = exps.sum();
        let output: ArrayD<f32> = exps / sum_of_exps;
        debug!("output shape: {:#?}", output.shape());
        debug!("output ndim: {:#?}", output.ndim());
        debug!("output slice 1x10: {:#?}", output.slice(s![.., 0..10]));
        output.into()
    }
}

#[no_mangle]
#[wasm_bindgen]
pub fn set_console_hook() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(Level::Debug).expect("failed init console log");
}
