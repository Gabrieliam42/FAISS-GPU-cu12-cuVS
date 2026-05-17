# FAISS-GPU-cu12-cuVS

Unofficial FAISS GPU + cuVS wheel build for Python 3.12 on Linux x86_64.

This repo now contains:

```text
wheels/faiss_gpu_cu12_cuvs-1.14.1.post1-cp312-cp312-manylinux_2_38_x86_64.whl
```

## Build Summary

| Property | Value |
|---|---|
| Package name | `faiss-gpu-cu12-cuvs` |
| Package version | `1.14.1.post1` |
| Wheel tag | `cp312-cp312-manylinux_2_38_x86_64` |
| FAISS version | `1.14.1` |
| Python | `3.12` |
| CUDA toolkit used for build | `12.9.1` |
| cuVS | enabled |
| RAPIDS cuVS runtime line | `25.10.0` |
| GPU target | `sm_86` Ampere native build |
| BLAS | OpenBLAS |
| Platform | Linux x86_64 / WSL2 |

## Verified Stack

This wheel was verified with the following stack:

```text
torch==2.9.1+cu129
torchaudio==2.9.1+cu129
torchvision==0.24.1+cu129
flash-attn==2.8.3
triton==3.5.1
Python 3.12
CUDA runtime line 12.9.1
```

Validation covered:

- import of `faiss`, `torch`, `flash_attn`, and `triton`
- `faiss.StandardGpuResources()`
- `faiss.GpuIndexFlatL2`
- GPU add/search smoke test
- presence of cuVS-facing symbols such as `GpuIndexCagra` and `IndexHNSWCagra`

## Compatibility

This specific wheel is tagged `manylinux_2_38_x86_64`.

That means the practical Linux/glibc floor is newer than Ubuntu 22.04. Treat this wheel as intended for:

- Ubuntu 24.04+
- WSL2 distros with glibc 2.38+

For older glibc baselines, this exact wheel is not the right artifact.

## Files

- Wheel: [wheels/faiss_gpu_cu12_cuvs-1.14.1.post1-cp312-cp312-manylinux_2_38_x86_64.whl](wheels/faiss_gpu_cu12_cuvs-1.14.1.post1-cp312-cp312-manylinux_2_38_x86_64.whl)

## Installation

Install into a Python 3.12 Linux environment:

```bash
pip install --extra-index-url https://pypi.nvidia.com \
  ./wheels/faiss_gpu_cu12_cuvs-1.14.1.post1-cp312-cp312-manylinux_2_38_x86_64.whl
```

If you want the exact CUDA-side dependency line aligned with the validated build, install these first:

```bash
pip install --extra-index-url https://pypi.nvidia.com \
  nvidia-cuda-runtime-cu12==12.9.79 \
  nvidia-cublas-cu12==12.9.1.4 \
  nvidia-curand-cu12==10.3.10.19 \
  nvidia-cusolver-cu12==11.7.5.82 \
  nvidia-cusparse-cu12==12.5.10.65 \
  nvidia-nvjitlink-cu12==12.9.86 \
  libcuvs-cu12==25.10.0 \
  libraft-cu12==25.10.0 \
  librmm-cu12==25.10.0 \
  rapids-logger==0.1.19
```

For the matching PyTorch stack:

```bash
pip install \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  torch==2.9.1+cu129 \
  torchaudio==2.9.1+cu129 \
  torchvision==0.24.1+cu129
```

FlashAttention and Triton used during validation:

```bash
pip install \
  "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
  "triton @ https://download.pytorch.org/whl/triton/triton-3.5.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
```

## Runtime Check

```python
import faiss
import numpy as np

res = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatL2(res, 4)

xb = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype="float32")

index.add(xb)
D, I = index.search(xb[:2], 2)
print(I)
print(D)
```

## Notes

- This is an unofficial wheel build, not an official Meta or NVIDIA distribution.
- The wheel includes cuVS-enabled FAISS support, which is the point of this repo.
- The build target is Ampere `sm_86`; newer architectures may work if the runtime path is compatible, but this repo is documenting the validated build, not claiming universal GPU coverage.

## License

FAISS is licensed under the [MIT License](https://github.com/facebookresearch/faiss/blob/main/LICENSE).
