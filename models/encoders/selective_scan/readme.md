# Selective scan CUDA extension

This package builds `selective_scan_cuda_core` and installs the `selective_scan`
Python interface used by InState-Mamba. Install the CUDA-enabled PyTorch build and
the repository's Python requirements first. CUDA Toolkit 11.8 (including
`nvcc`) and a compatible C++ compiler are required for the documented setup.

From the repository root:

```bash
python -m pip install --no-build-isolation ./models/encoders/selective_scan
python -c "import torch, selective_scan, selective_scan_cuda_core; print('Selective scan import succeeded')"
```

The extension is compiled for the local PyTorch and CUDA environment; build it
again if that environment changes. Do not copy a compiled `.so` from an
unrelated environment.
