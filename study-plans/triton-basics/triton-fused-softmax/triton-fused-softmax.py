import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, x_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * x_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + col_offsets
    mask = col_offsets < n_cols

    this_row = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    stable_row = tl.exp(this_row - tl.max(this_row, axis=0))
    this_row_softmax = stable_row / tl.sum(stable_row, axis=0)

    out_block_start = pid * out_row_stride + col_offsets
    tl.store(out_ptr + out_block_start, this_row_softmax, mask=mask)
    


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch softmax_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    softmax_kernel[grid](
        x, out, x.stride(0), out.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )