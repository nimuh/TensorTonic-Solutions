import torch
import triton
import triton.language as tl


@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, x_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * x_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + col_offsets
    mask = col_offsets < n_cols

    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    row_max = tl.max(x, axis=0)
    log_row_sum = tl.log( tl.sum(tl.exp(x - row_max), axis=0) ) + row_max

    tl.store(out_ptr + pid, log_row_sum)
    
    
def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch logsumexp_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    logsumexp_kernel[grid](
        x, out, x.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )