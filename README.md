# FAISS-GPUu-cuVS — Unofficial pip Wheel

**FAISS 1.14.0** built from source with full GPU and NVIDIA cuVS support.

This fills a gap that does not exist elsewhere:
- `faiss-gpu-cu12` on PyPI (including 1.14.0) — GPU only, **no cuVS**
- `faiss-gpu-cuvs` official — **conda only**, not on PyPI

---

## Wheel

```
faiss_gpu_cu12_cuvs-1.14.0-cp312-cp312-linux_x86_64.whl
```

| Property | Value |
|---|---|
| FAISS version | 1.14.0 |
| Python | 3.12 |
| CUDA | 12.x |
| GPU architecture | SM 86 (RTX 3090 / RTX 3080) |
| cuVS | Enabled (`FAISS_ENABLE_CUVS=ON`) |
| CPU SIMD | AVX2 |
| BLAS | OpenBLAS |
| Platform | Linux x86_64 |

> **GPU Architecture Compatibility:**
> - **SM 86 — native:** RTX 3080, RTX 3090, RTX 3090 Ti, A40
> - **SM 89, 90 — works via PTX JIT:** RTX 4000 series, H100 (slower on first run, then cached)
> - **SM 80 and below — will not work:** A100, RTX 2000 series, V100 and older
>
> To support all architectures, rebuild from source with `-DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90"`.

---

## Installation

### 1. Install RAPIDS dependencies

```bash
pip install \
  libcuvs-cu12==25.10.0 \
  librmm-cu12==25.10.0 \
  libraft-cu12==25.10.0 \
  rapids-logger \
  "nvidia-nvjitlink-cu12>=12.9" \
  --extra-index-url https://pypi.nvidia.com
```

### 2. Install system dependency

```bash
sudo apt-get install -y libopenblas-dev
```

### 3. Install the wheel

```bash
pip install faiss_gpu_cu12_cuvs-1.14.0-cp312-cp312-linux_x86_64.whl
```

---

## Verification

```python
import faiss

res = faiss.StandardGpuResources()
idx_cpu = faiss.IndexFlatIP(128)

opt = faiss.GpuClonerOptions()
opt.useFloat16 = True
opt.use_cuvs = True  # cuVS enabled

idx_gpu = faiss.index_cpu_to_gpu(res, 0, idx_cpu, opt)
print("cuVS GPU index ready:", idx_gpu.ntotal == 0)
```

---

## Runtime Requirements

| Package | Version | Source |
|---|---|---|
| libcuvs-cu12 | 25.10.0 | pypi.nvidia.com |
| librmm-cu12 | 25.10.0 | pypi.nvidia.com |
| libraft-cu12 | 25.10.0 | pypi.nvidia.com |
| rapids-logger | 0.1.19+ | pypi.nvidia.com |
| nvidia-nvjitlink-cu12 | 12.9.86+ | PyPI |
| libopenblas-dev | any | apt |
| CUDA runtime | 12.x | system |

---

## Build Environment

| Component | Version |
|---|---|
| FAISS source | v1.14.0 (facebookresearch/faiss) |
| nvcc | 12.0.140 |
| gcc | 13.3.0 |
| cmake | 4.2.3 |
| Python | 3.12.3 |
| OS | Linux (WSL2 / Ubuntu) |

### CMake flags used

```
-DFAISS_ENABLE_GPU=ON
-DFAISS_ENABLE_CUVS=ON
-DFAISS_ENABLE_PYTHON=ON
-DFAISS_OPT_LEVEL=avx2
-DCMAKE_CUDA_ARCHITECTURES=86
-DBLA_VENDOR=OpenBLAS
-DCMAKE_BUILD_TYPE=Release
-DBUILD_TESTING=OFF
-DBUILD_SHARED_LIBS=ON
```

---

## Why does this exist?

`faiss-gpu-cuvs` is officially distributed via conda only. No pip wheel exists.
The standard `faiss-gpu-cu12` pip package does not enable cuVS even at version 1.14.0.
This wheel is a self-contained pip-installable build of FAISS 1.14.0 with cuVS enabled.

---

## License

FAISS is licensed under the [MIT License](https://github.com/facebookresearch/faiss/blob/main/LICENSE).
This wheel is an unofficial build. Not affiliated with Meta or NVIDIA.
