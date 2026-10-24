# Origin of this backbone

This folder is the **Spherical Voronoi** code GoDe was developed and evaluated with, modified for GoDe. It is kept here as is, rasterizer sources included, so that the release does not depend on the upstream repositories.

- Upstream: https://github.com/sphericalvoronoi/sphericalvoronoi (MIT, see the licence files in this folder).
- Added for GoDe: the backbone interface in the model class, `scalable_train.py`, and `gode/`, a link to the shared core at the top of the repository.
- Beta kernels with Spherical Voronoi colour; based on the Spherical Voronoi code release at commit `f167e90` of the upstream repository.
- `submodules/gsplat` is gsplat (nerfstudio, Apache-2.0) as modified by [Beta Splatting](https://github.com/RongLiu-Leo/beta-splatting) (`rasterization()` takes `betas`).
- `submodules/spherical-voronoi` is the Spherical Voronoi colour CUDA extension (MIT).
- The bundled GLM (`third_party/glm`) keeps its headers and licence; its documentation and tests, not needed to build, are left out.

Tested with Python 3.10.12, PyTorch 2.5.1 and CUDA 12.1 (versions in `requirements.txt`).
