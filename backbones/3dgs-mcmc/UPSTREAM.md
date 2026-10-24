# Origin of this backbone

This folder is the **3DGS-MCMC** code GoDe was developed and evaluated with, modified for GoDe. It is kept here as is, rasterizer sources included, so that the release does not depend on the upstream repositories.

- Upstream: https://github.com/ubc-vision/3dgs-mcmc (Gaussian-Splatting License, see the licence files in this folder).
- Added for GoDe: the backbone interface in the model class, `scalable_train.py`, and `gode/`, a link to the shared core at the top of the repository.
- Quantization kept as in the paper: geometry float16, SH int8.
- `submodules/diff-gaussian-rasterization`: shakibakh/diff-gaussian-rasterization @ `1761eb9` (branch `gs-mcmc`, the commit upstream 3DGS-MCMC pins), CUDA unchanged; the Python package is renamed `mcmc_diff_gaussian_rasterization` so it can share an environment with other 3DGS rasterizers.
- `submodules/simple-knn`: Inria simple-knn @ `f155ec0`, unchanged.
- The bundled GLM (`third_party/glm`) keeps its headers and licence; its documentation and tests, not needed to build, are left out.

Tested with Python 3.10.12, PyTorch 2.5.1 and CUDA 12.1 (versions in `requirements.txt`).
