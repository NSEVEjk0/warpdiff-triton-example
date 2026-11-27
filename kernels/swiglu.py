"""Apply SiLU, then multiply by the value in a separate kernel."""

import triton
import triton.language as tl


@triton.jit
def _silu(gate, scratch, size: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    valid = offsets < size
    x = tl.load(gate + offsets, mask=valid, other=0)
    tl.store(scratch + offsets, x / (1.0 + tl.exp(-x)), mask=valid)


@triton.jit
def _multiply(scratch, value, output, size: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    valid = offsets < size
    activation = tl.load(scratch + offsets, mask=valid, other=0)
    y = tl.load(value + offsets, mask=valid, other=0)
    tl.store(output + offsets, activation * y, mask=valid)


def swiglu(gate, value, output, scratch):
    size = gate.numel()
    block = 1024
    grid = (triton.cdiv(size, block),)
    _silu[grid](gate, scratch, size, block)
    _multiply[grid](scratch, value, output, size, block)
