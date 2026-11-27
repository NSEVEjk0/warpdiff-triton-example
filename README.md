# WarpDiff Triton example

This repo uses the public WarpDiff GitHub Action to test a Triton SwiGLU change on a Vast.ai RTX 3090. The benchmark checks `silu(gate) * value` against PyTorch and times the original and patched kernels.

- [Benchmark definition](warpdiff.toml)
- [PyTorch reference and timing adapter](benchmarks/swiglu.py)
- [GitHub Actions workflow](.github/workflows/warpdiff.yml)
- [WarpDiff setup guide](https://github.com/NSEVEjk0/warpdiff#use-warpdiff-in-another-repo)

To reproduce it, copy these files into a repo you control. Put the original kernel on the default branch and open a pull request with the fused kernel. Add a scoped Vast.ai key as the `VASTAI_API_KEY` Actions secret, then open **Actions → GPU benchmark → Run workflow** and enter the PR number. The workflow uses the default benchmark when the benchmark input is blank. The **Checks** tab on the pull request shows the result.
