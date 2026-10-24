<div align="center">

# GoDe: Gaussians on Demand for Backbone-Agnostic Progressive Compression

**WACV 2027**

[Francesco Di Sario](https://scholar.google.com/citations?user=wjOMl_8AAAAJ)<sup>1,2</sup> ·
[Riccardo Renzulli](https://scholar.google.com/citations?user=JlAby_oAAAAJ)<sup>1</sup> ·
[Marco Grangetto](https://scholar.google.com/citations?user=Pt1gmQYAAAAJ)<sup>1</sup> ·
[Akihiro Sugimoto](https://scholar.google.com/citations?user=r7ZaWDUAAAAJ)<sup>2</sup> ·
[Enzo Tartaglione](https://scholar.google.com/citations?user=uKuvN64AAAAJ)<sup>3</sup>

<sup>1</sup>University of Turin · <sup>2</sup>National Institute of Informatics, Tokyo · <sup>3</sup>Télécom Paris, Institut Polytechnique de Paris

[![Project page](https://img.shields.io/badge/Project-page-111111)](https://gaussians-on-demand.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2501.13558-b31b1b)](https://arxiv.org/abs/2501.13558)
[![WACV 2027](https://img.shields.io/badge/WACV-2027-8c6c20)](https://arxiv.org/abs/2501.13558)
[![License](https://img.shields.io/badge/core-MIT-lightgrey)](LICENSE)

</div>

![One hierarchical GoDe model of bicycle decoded at levels 0, 1, 2 and 7](assets/teaser.png)

<p align="center"><sub>One hierarchical 3DGS-MCMC model of <i>bicycle</i>: each level of detail adds a layer of primitives and is
decoded from a longer prefix of the same compressed file. PSNR on the test views; FPS of this view at 1080p on an NVIDIA A40.</sub></p>

GoDe turns a trained splat-based radiance field into **one progressive, compressed bitstream**. It ranks the
primitives by gradient importance into nested levels of detail, fine-tunes all levels at once with
quantization-aware training, and stores a base layer and enhancement layers as chunks compressed with shared
Zstandard dictionaries. Any prefix of the file decodes to a complete model: one model covers the whole
rate–distortion range, with no retraining per rate, and the same method runs unchanged on Gaussians, triangles,
neural anchors, octrees and Beta kernels.

## Highlights

- **One model, every rate.** On *bicycle*, a single 3DGS-MCMC model goes from 6.1 MB (level 0) to 101.5 MB
  (level 7); the model it starts from is 1.44 GB uncompressed.
- **Backbone-agnostic.** The same core runs on five backbones — 3DGS-MCMC, Scaffold-GS, Octree-GS, Triangle
  Splatting and Spherical Voronoi — without modifying them.
- **Fast progressive decoding.** Decoding a whole Mip-NeRF 360 scene, every level included, takes 0.5–1.3 s,
  against 9–88 s for single-rate codecs and 42–52 s for the progressive PCGS.
- **Competitive quality.** Rate–distortion on par with single-rate codecs, which train a separate model for every
  rate.

## Method

![The GoDe pipeline](assets/pipeline.png)

1. **Hierarchy of splats.** Primitives are ranked by their accumulated gradients and the least important are
   masked out level by level, recomputing the gradients after every step. Every level is a subset of the next:
   a base layer 𝒢₀ and enhancement layers ℰ₁ … ℰ_{L−1}. Building eight levels takes 1–2 minutes.
2. **One quantization-aware fine-tuning.** All levels share one set of parameters. Each iteration samples a level
   and a view and renders that level with simulated quantization (float16 geometry, int8 appearance), so the cost
   does not grow with the number of levels.
3. **Progressive coding.** Each layer is a chunk compressed with shared Zstandard dictionaries. Decoding a level
   means decoding and concatenating its chunks; an opacity ramp makes switching between levels smooth.

## Results

**Levels on Mip-NeRF 360** — coarsest and finest of the eight levels, dataset averages (supplementary Table S7;
FPS on an NVIDIA A40):

| Backbone | Level 0: PSNR · size · FPS | Level 7: PSNR · size · FPS |
|---|---|---|
| 3DGS-MCMC | 26.97 dB · 11.3 MB · 270 | 27.69 dB · 61.5 MB · 122 |
| Scaffold-GS | 27.29 dB · 9.8 MB · 104 | 27.93 dB · 41.6 MB · 68 |
| Octree-GS | 26.75 dB · 11.2 MB · 66 | 27.84 dB · 36.2 MB · 33 |
| Triangle Splatting | 24.68 dB · 9.8 MB · 203 | 26.42 dB · 87.9 MB · 57 |
| Spherical Voronoi | 27.53 dB · 10.0 MB · 213 | 28.38 dB · 65.3 MB · 80 |

**Training and decoding time** on Mip-NeRF 360 (paper, Table 3). Single-rate codecs train once per rate; GoDe
trains once for all of them. Decoding is in seconds at the low / medium / high operating points.

| Method | Progressive | Training, all rates (h) | Decoding (s) |
|---|:-:|:-:|:-:|
| Context-GS | | 4.0 | 53.1 / 82.5 / 88.1 |
| HAC++ | | 3.9 | 12.3 / 13.4 / 19.0 |
| RDO | | 3.9 | 2.2 / 7.4 / 19.0 |
| Comp-GS | | 4.9 | 9.1 / 9.0 / 8.8 |
| PCGS | ✓ | 1.8 | 42.3 / 46.3 / 52.1 |
| **GoDe (3DGS)** | ✓ | **1.1** | **0.6 / 0.9 / 1.3** |
| **GoDe (Scaffold-GS)** | ✓ | **0.6** | **0.2 / 0.3 / 0.5** |

## Repository layout

| Path | Contents |
|---|---|
| [`gode/`](gode) | The method: hierarchy, quantization-aware fine-tuning, progressive coding, metrics. No backbone code; its tests run on CPU. |
| [`backbones/3dgs-mcmc`](backbones/3dgs-mcmc) | 3D Gaussians trained with MCMC, spherical-harmonics colour |
| [`backbones/scaffold-gs`](backbones/scaffold-gs) | Scaffold-GS, neural anchors |
| [`backbones/octree-gs`](backbones/octree-gs) | Octree-GS, anchors in a level-of-detail octree |
| [`backbones/triangle-splatting`](backbones/triangle-splatting) | Triangle Splatting |
| [`backbones/spherical-voronoi`](backbones/spherical-voronoi) | Beta kernels with Spherical Voronoi colour |

Each backbone is the code used for the paper, CUDA rasterizers included, so nothing is fetched from other
repositories. Each one has its own README (installation, data, training, evaluation), the licence of the project it
derives from, an `UPSTREAM.md` saying where it comes from and what changed, and a pinned `requirements.txt`
(Python 3.10, PyTorch 2.5.1, CUDA 12.1). Its `gode/` is a link to the shared core.

## Getting started

**1. Clone, and pick a backbone.** Backbones need separate environments (Octree-GS needs the stock `gsplat`,
Spherical Voronoi the Beta Splatting variant); each README gives the exact commands.

```shell
git clone https://github.com/gaussians-on-demand/GoDe.git
cd GoDe/backbones/3dgs-mcmc
conda create -y -n gode-mcmc python=3.10 && conda activate gode-mcmc
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization submodules/simple-knn
```

**2. Get the data.** The standard benchmarks, in COLMAP layout:

| Dataset | Download |
|---|---|
| Mip-NeRF 360 | [jonbarron.info/mipnerf360](https://jonbarron.info/mipnerf360) |
| Tanks & Temples, Deep Blending | [tandt_db.zip](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip) |

```
$DATA_ROOT/nerf_real_360/{bicycle,garden,...}
$DATA_ROOT/tandt/{train,truck}
$DATA_ROOT/db/{drjohnson,playroom}
```

**3. Run GoDe** in two stages: train the backbone (or bring your own checkpoint), then build and compress the
hierarchy.

```shell
python train.py --config configs/bonsai.json --eval --images images_2 \
  -s $DATA_ROOT/nerf_real_360/bonsai -m output/bonsai

python scalable_train.py --config configs/bonsai.json --images images_2 \
  -s $DATA_ROOT/nerf_real_360/bonsai -m output/gode/bonsai \
  --checkpoint output/bonsai/point_cloud/iteration_30000/point_cloud.ply \
  --num_levels 8 --hier_min 150000 --sample_level_ratio 1.1 \
  --ft_iterations 30000 --save_images
```

The output holds the compressed chunks `lod_0.zst … lod_7.zst` with their shared dictionary, and
`scalable_results.csv` with PSNR, SSIM, LPIPS, primitives, size, FPS and decoding time for every level.
Each backbone's README shows how to run whole benchmarks, and `tools/make_results_table.py` aggregates the results.

The core's tests need neither a GPU nor a dataset:

```shell
pytest gode/tests
```

## Adding a backbone

`gode/` never touches primitives directly. A backbone is any model exposing:

| Group | Methods |
|---|---|
| Rendering | `render(camera, mask=None, measure_fps=False)` returning `{"render": ...}` |
| Optimisation | `training_setup(args)`, `update_learning_rate(it)`, `optimizer`, `train()`, `eval()` |
| Primitives | `num_primitives`, `primitive_names()`, `index_primitives(idx)`, `subset(mask)`, `append_primitives(other)`, `clone_empty()` |
| Importance | `_get_gradient_scores()`, `gradient_prune(k, prune_type, scores)` |
| Objective | `photometric_loss(image, gt, lambda_dssim)`, `appearance_regularization(mask, weight)`, `geometry_groups()` |
| Compression | `_sort_morton()`, `set_qa()`, `quantizer_names()`, `calibrate_quantizers_for_compression()`, `encode_payload(quantize)`, `decode_payload(payload)` |

`encode_payload` returns `{"geom": {...}, "features": {...}}`, where arrays are numpy and everything else is
metadata; `gode/streams.py` handles dictionary training, the container format and incremental decoding. Adding a
backbone therefore means implementing these methods on its model class and pointing a copy of `scalable_train.py`
at it: the hierarchy, the fine-tuning schedule and the codec stay the same. See
[`backbones/spherical-voronoi/scene/beta_model.py`](backbones/spherical-voronoi/scene/beta_model.py) for a reference
implementation.

## Citation

```bibtex
@article{disario2026gode,
  title   = {GoDe: Gaussians on Demand for Backbone-Agnostic Progressive Compression},
  author  = {Di Sario, Francesco and Renzulli, Riccardo and Grangetto, Marco and
             Sugimoto, Akihiro and Tartaglione, Enzo},
  journal = {arXiv preprint arXiv:2501.13558},
  year    = {2026}
}
```

## Acknowledgements

GoDe builds on the code of
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting),
[3DGS-MCMC](https://github.com/ubc-vision/3dgs-mcmc),
[Scaffold-GS](https://github.com/city-super/Scaffold-GS),
[Octree-GS](https://github.com/city-super/Octree-AnyGS),
[Triangle Splatting](https://github.com/trianglesplatting/triangle-splatting),
[Beta Splatting](https://github.com/RongLiu-Leo/beta-splatting),
[Spherical Voronoi](https://github.com/sphericalvoronoi/sphericalvoronoi),
[gsplat](https://github.com/nerfstudio-project/gsplat) and
[lpips-pytorch](https://github.com/S-aiueo32/lpips-pytorch). We thank their authors for releasing them.

## License

The GoDe core (`gode/`) is released under the [MIT licence](LICENSE). Each folder under `backbones/` keeps the
licence of the project it derives from (see the licence files and `UPSTREAM.md` in each folder); several of them
are under the Gaussian-Splatting License, which allows research and evaluation use only.
