use crate::matrix::Matrix;
use metal::MTLResourceOptions;
use std::mem;

use super::gpu;

const LIB_DATA: &[u8] = include_bytes!("../kernel/matmul.metallib");

/// Parallel computation of a matrix multiplication.
/// Only admits square matrices.
pub fn prod<T: Copy>(
    ma: &Matrix<T>,
    mb: &Matrix<T>,
    state: &mut gpu::MetalState,
) -> *mut std::ffi::c_void {
    assert!(ma.is_square());
    assert!(mb.is_square());
    assert_eq!(ma.rows, mb.rows);
    let size = ma.sizeof_entries();
    println!("Size mem: {:?}", size);

    let buffer_a = state.device.new_buffer_with_data(
        unsafe { mem::transmute(ma.entries.as_ptr()) },
        size,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_b = state.device.new_buffer_with_data(
        unsafe { mem::transmute(mb.entries.as_ptr()) },
        size,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_result = state.device.new_buffer(
        size, // the result will be another square matrix of the same size
        MTLResourceOptions::StorageModeShared,
    );

    let lib = state.device.new_library_with_data(LIB_DATA).unwrap();
    let function = lib.get_function("mul_matrices", None).unwrap();
    let pipeline = state
        .device
        .new_compute_pipeline_state_with_function(&function)
        .unwrap();

    let n = ma.rows as u64;
    let w = pipeline.thread_execution_width();
    let h = pipeline.max_total_threads_per_threadgroup() / w;
    state.grid_size = metal::MTLSize::new(n, n, 1);
    state.threadgroup_size = metal::MTLSize::new(w, h, 1);

    gpu::execute(
        function,
        &[Some(&buffer_a), Some(&buffer_b), Some(&buffer_result)],
        &[0; 3],
        state,
    );

    buffer_result.contents()
}
