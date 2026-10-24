# Origin of this backbone

This folder is the **Triangle Splatting** code GoDe was developed and evaluated with, modified for GoDe. It is kept here as is, rasterizer sources included, so that the release does not depend on the upstream repositories.

- Upstream: https://github.com/trianglesplatting/triangle-splatting (Apache-2.0 and the Gaussian-Splatting License, see the licence files in this folder).
- Added for GoDe: the backbone interface in the model class, `scalable_train.py`, and `gode/`, a link to the shared core at the top of the repository.
- Based on upstream commit `685ff27`; the upstream DTU/TNT mesh evaluation is removed.
- `submodules/diff-triangle-rasterization`: trianglesplatting/diff-triangle-rasterization @ `6d61f4c`, unchanged.
- `submodules/simple-knn`: Inria simple-knn @ `86710c2`, unchanged.
- The bundled GLM (`third_party/glm`) keeps its headers and licence; its documentation and tests, not needed to build, are left out.

Tested with Python 3.10.12, PyTorch 2.5.1 and CUDA 12.1 (versions in `requirements.txt`).
