use metal::{Device, DeviceRef};

mod dot;
pub mod gpu;
mod matmul;
pub mod matrix;
pub mod utils;

fn main() {
    // will return a raw pointer to the result
    // the system will assign a GPU to use.
    let device: &DeviceRef = &Device::system_default().expect("No device found");
    let queue = device.new_command_queue();

    let state = &mut gpu::MetalState {
        device,
        queue,
        grid_size: Default::default(),
        threadgroup_size: Default::default(),
    };

    let result = dot::dotprod(&[3, 4, 1, 7, 10, 20], &[2, 5, 6, 9, 5, 10], state);
    println!("Dot product result: {:?}", result);

    let matrix_a = matrix::Matrix::new(4, 4, &[1f32; 16]);
    let matrix_b = matrix::Matrix::new(4, 4, &[2f32; 16]);

    let result = matmul::prod(&matrix_a, &matrix_b, state) as *const [f32; 16];

    unsafe {
        println!("Matrix A: {:?}", matrix_a);
        println!("Matrix B: {:?}", matrix_b);
        println!("A x B result: {:?}", *result);
    };
}
