use metal::{BufferRef, CommandQueue, Function, MTLSize, NSUInteger};

pub struct MetalState<'a> {
    pub device: &'a metal::DeviceRef,
    pub queue: CommandQueue,
    pub grid_size: MTLSize,
    pub threadgroup_size: MTLSize,
}

pub fn execute(
    function: Function,
    data: &[Option<&BufferRef>],
    offsets: &[NSUInteger],
    state: &MetalState,
) {
    // a command queue for sending instructions to the device.
    let command_queue = state.device.new_command_queue();
    // for sending commands, a command buffer is needed.
    let command_buffer = command_queue.new_command_buffer();
    // to write commands into a buffer an encoder is needed, in our case a compute encoder.
    let compute_encoder = command_buffer.new_compute_command_encoder();
    let pipeline = state
        .device
        .new_compute_pipeline_state_with_function(&function)
        .unwrap();
    compute_encoder.set_compute_pipeline_state(&pipeline);
    compute_encoder.set_buffers(0, data, offsets);

    println!(
        "Grid size: {:?} x {:?} x {:?}",
        state.grid_size.width, state.grid_size.height, state.grid_size.depth
    );
    println!(
        "TG size: {:?} x {:?} x {:?}",
        state.threadgroup_size.width, state.threadgroup_size.height, state.threadgroup_size.depth
    );

    // specify thread count and organization
    compute_encoder.dispatch_threads(state.grid_size, state.threadgroup_size);

    // end encoding and execute commands
    compute_encoder.end_encoding();
    command_buffer.commit();

    command_buffer.wait_until_completed();
}
