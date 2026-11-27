"""Generate inputs and a PyTorch reference for the SwiGLU kernel."""

import torch
from torch.nn import functional as F


def prepare(case):
    gate = torch.randn(case["size"], device="cuda", dtype=torch.float32)
    value = torch.randn_like(gate)
    expected = F.silu(gate) * value

    def bind(variant):
        output = torch.empty_like(expected)
        scratch = torch.empty_like(expected)

        def run():
            variant.swiglu(gate, value, output, scratch)

        return run, output

    return expected, bind
