# Origin of this backbone

This folder is the **Octree-GS** code GoDe was developed and evaluated with, modified for GoDe. It is kept here as is, rasterizer sources included, so that the release does not depend on the upstream repositories.

- Upstream: https://github.com/city-super/Octree-AnyGS (Gaussian-Splatting License, see the licence files in this folder).
- Added for GoDe: the backbone interface in the model class, `scalable_train.py`, and `gode/`, a link to the shared core at the top of the repository.
- This is the Octree-AnyGS version GoDe was developed on, older than the current upstream; it is kept as is.
- Needs the stock `gsplat==1.3.0` (pinned in `requirements.txt`); the upstream `environment.yml` installed an unpinned gsplat fork from git and is replaced by `requirements.txt`.
- Local paths removed: the default output folder is `./output/` as upstream, the configs' default `source_path` is `data/nerf_real_360/bicycle` (always overridden by `--source_path`), and a few commented-out debug paths are gone.

Tested with Python 3.10.12, PyTorch 2.5.1 and CUDA 12.1 (versions in `requirements.txt`).
