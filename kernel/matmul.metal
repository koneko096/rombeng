[[kernel]]
void mul_matrices(
  device const float* in_a [[buffer(0)]],
  device const float* in_b [[buffer(1)]],
  device float* result [[buffer(2)]],
  uint2 grid_size [[threads_per_grid]], // matrices size (row and col)
  uint2 pos [[thread_position_in_grid]])
{
  uint n = grid_size.x;
  result[pos.y * n + pos.x] = 0;

  for (uint i = 0; i < n; i++) {
    result[pos.y * n + pos.x] += in_a[pos.y * n + i] * in_b[i * n + pos.x];
  }
}