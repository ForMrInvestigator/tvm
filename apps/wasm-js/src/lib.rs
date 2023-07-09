#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate serde_derive;
extern crate console_error_panic_hook;

mod types;

use std::{collections::HashMap, convert::TryFrom};
use std::cmp::min;


use std::convert::TryInto;
use std::sync::Once;

use image::{DynamicImage, RgbaImage};
use image::imageops::FilterType;

use ndarray::prelude::*;
use ndarray::{Array, array};

use tvm_graph_rt::{Graph, GraphExecutor, SystemLibModule, Tensor as TVMTensor};
use wasm_bindgen::prelude::*;

use types::Tensor;

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    // The `console.log` is quite polymorphic, so we can bind it with multiple
    // signatures. Note that we need to use `js_name` to ensure we always call
    // `log` in JS.
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn log_u32(a: u32);

    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn log_vu8(a: Vec<u8>);

    // Multiple arguments too!
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn log_many(a: &str, b: &str);
}

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
    ONCE.call_once(|| unsafe {
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
        log("exec init");
    });

    // tensor to ndarray to image;
    let input: ArrayD<f32> = tensor.into();
    let mut image: DynamicImage = RgbaImage::from_vec(
        input.shape()[0] as u32, input.shape()[1] as u32,
        input.mapv(|x| x as u8).into_raw_vec(),
    ).unwrap().into();

    // center crop and scale and to rgb
    let scale = 224;
    let crop_size = min(image.height(), image.width());
    let rgb = image.crop(
        (image.width() - crop_size) / 2,
        (image.height() - crop_size) / 2,
        crop_size, crop_size,
    ).resize(scale, scale, FilterType::Nearest).to_rgb8();

    // to ndarray again
    let input = Array::from_shape_vec([scale as usize, scale as usize, 3], rgb.to_vec()).unwrap();
    let mut input = input.mapv(|x| x as f32);

    // normalization
    // input * radio - mean * reverse_std
    let mean = [0.485, 0.456, 0.406];
    let reverse_div = [4.367, 4.464, 4.444];
    let ratio = 1.0 / 255.0;
    let normalized_div = array![ratio * reverse_div[0], ratio * reverse_div[1], ratio * reverse_div[2]];
    let normalized_mean = array![mean[0] * reverse_div[0], mean[1] * reverse_div[1], mean[2] * reverse_div[2]];
    input = input * normalized_div - normalized_mean;

    //HWC to CHW
    input = input.permuted_axes([2, 0, 1]);

    //clone items to make the memory layout contiguous;
    let contiguous_input = Array::from_shape_vec(input.raw_dim(), input.iter().cloned().collect()).unwrap();
    unsafe {
        // set input tensor and run the graph get output;
        let mut executor = EXEC.take().unwrap();
        executor.set_input("input", contiguous_input.into());
        executor.run();
        let output: ArrayD<f32> = executor.get_output(0).unwrap().to_owned().try_into().unwrap();
        EXEC.replace(executor);
        // softmax;
        let exps = output.map(|x| x.exp());
        let sum_of_exps = exps.sum();
        let output = exps / sum_of_exps;
        output.into()
    }
}


#[wasm_bindgen]
pub fn set_console_hook() {
    console_error_panic_hook::set_once();
}
