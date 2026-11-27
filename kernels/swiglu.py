"""Apply SiLU and multiply by the value in one GPU pass."""

import triton
import triton.language as tl


@triton.jit
def _swiglu(gate, value, output, size: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    valid = offsets < size
    x = tl.load(gate + offsets, mask=valid, other=0)
    y = tl.load(value + offsets, mask=valid, other=0)
    activation = x / (1.0 + tl.exp(-x))
    tl.store(output + offsets, activation * y, mask=valid)


def swiglu(gate, value, output, _scratch):
    size = gate.numel()
    block = 1024
    _swiglu[(triton.cdiv(size, block),)](gate, value, output, size, block)
