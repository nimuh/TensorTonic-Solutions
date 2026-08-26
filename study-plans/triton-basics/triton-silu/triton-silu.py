import torch
import triton
import triton.language as tl


@triton.jit
def silu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # set up
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)
    out = x / (1 + tl.exp(-1 * x))

    
    tl.store(out_ptr + offsets, out, mask=mask)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch silu_kernel: out = x / (1 + exp(-x))."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    silu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)