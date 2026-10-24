# Origin of this backbone

This folder is the **Scaffold-GS** code GoDe was developed and evaluated with, modified for GoDe. It is kept here as is, rasterizer sources included, so that the release does not depend on the upstream repositories.

- Upstream: https://github.com/city-super/Scaffold-GS (Gaussian-Splatting License, see the licence files in this folder).
- Added for GoDe: the backbone interface in the model class, `scalable_train.py`, and `gode/`, a link to the shared core at the top of the repository.
- `submodules/scaffold-diff-gaussian-rasterization` and `submodules/simple-knn`: as shipped by upstream Scaffold-GS, CUDA unchanged; the rasterizer's Python package is renamed `scaffold_diff_gaussian_rasterization`.
- The upstream `environment.yml` (unpinned) is replaced by `requirements.txt`.
- The bundled GLM (`third_party/glm`) keeps its headers and licence; its documentation and tests, not needed to build, are left out.

Tested with Python 3.10.12, PyTorch 2.5.1 and CUDA 12.1 (versions in `requirements.txt`).
