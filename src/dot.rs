use core::slice;
use std::mem;

use crate::gpu::MetalState;
use metal::{MTLResourceOptions, MTLSize};

use super::gpu;

const LIB_DATA: &[u8] = include_bytes!("../kernel/dotproduct.metallib");

pub fn dotprod(v: &[u32], w: &[u32], state: &mut MetalState) -> Vec<u32> {
    // represents the library which contains the kernel.
    let lib = state.device.new_library_with_data(LIB_DATA).unwrap();
    // create function pipeline.
    // this compiles the function, so a pipeline can't be created in performance sensitive code.
    let function = lib.get_function("dot_product", None).unwrap();

    let length = v.len() as u64;
    let size = length * mem::size_of::<u32>() as u64;
    assert_eq!(v.len(), w.len());

    let buffer_a = state.device.new_buffer_with_data(
        unsafe { mem::transmute(v.as_ptr()) },
        size,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_b = state.device.new_buffer_with_data(
        unsafe { mem::transmute(w.as_ptr()) },
        size,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_result = state.device.new_buffer(
        size, // the operation will return an array with the same size.
        MTLResourceOptions::StorageModeShared,
    );

    state.grid_size = MTLSize::new(length, 1, 1);
    state.threadgroup_size = MTLSize::new(length, 1, 1);

    gpu::execute(
        function,
        &[Some(&buffer_a), Some(&buffer_b), Some(&buffer_result)],
        &[0; 3],
        state,
    );

    let ptr = buffer_result.contents() as *const u32;
    let len = buffer_result.length() as usize / mem::size_of::<u32>();
    let slice = unsafe { slice::from_raw_parts(ptr, len) };
    slice.to_vec()
}
