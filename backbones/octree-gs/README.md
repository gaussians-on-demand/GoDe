# GoDe: Gaussians on Demand for Backbone-Agnostic Progressive Compression

Francesco Di Sario, Riccardo Renzulli, Marco Grangetto, Akihiro Sugimoto, Enzo Tartaglione

[Project page](https://gaussians-on-demand.github.io/) · [arXiv](https://arxiv.org/abs/2501.13558)

GoDe turns a pre-trained radiance field into a **progressive, level-of-detail
bitstream**. A gradient-based importance criterion organises the primitives
into nested levels, all levels are fine-tuned jointly under quantization-aware
training, and the levels are stored as disjoint chunks sharing a trained zstd
dictionary. A client decodes chunk 0 to display the coarsest model and keeps
appending chunks to refine it, without ever re-downloading what it already has.

This folder is the **Octree-GS** backbone of the GoDe repository ([overview](../../README.md)). The method core is [`gode/`](../../gode), shared by all five backbones.

---

## Installation

```shell
conda create -y -n gode-octree python=3.10
conda activate gode-octree
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install -r requirements.txt
```

> **This backbone needs the stock `gsplat`.** Spherical Voronoi ships a fork of
> gsplat whose `rasterization()` takes an extra `betas` argument, so the two
> cannot share one environment: installing that fork here raises
> `TypeError: rasterization() missing 1 required positional argument: 'betas'`.
> Keep Octree-GS in its own environment, or put the stock package first on
> `PYTHONPATH`.

## Data

The standard benchmarks are expected in COLMAP layout, and `$DATA_ROOT` points
at the directory that contains them:

```
$DATA_ROOT/nerf_real_360/{bicycle,garden,...}
$DATA_ROOT/tandt/{train,truck}
$DATA_ROOT/db/{drjohnson,playroom}
```

As in the MipNeRF-360 protocol, indoor scenes are evaluated on `images_2` and
outdoor ones on `images_4`; do not downsample with `-r`. Anchor hyper-parameters
(`--voxel_size`, `--update_init_factor`, `--appearance_dim`) keep the Scaffold-GS
defaults; pass the per-dataset values from that paper when reproducing it.

## Usage

GoDe runs in two stages.

**1. Pre-train the backbone** (or bring your own checkpoint):

```shell
python train.py --config config/scaffoldgs/lod_model.yaml \
  --source_path $DATA_ROOT/nerf_real_360/bicycle \
  --scene_name bicycle --dataset_name mipnerf360
```

**2. Build and compress the hierarchy:**

```shell
python scalable_train.py --config config/scaffoldgs/lod_model.yaml \
  --source_path $DATA_ROOT/nerf_real_360/bicycle \
  --model_path output/gode/bicycle \
  --checkpoint output/bicycle/point_cloud/iteration_40000 \
  --num_levels 8 --hier_min 100000 --sample_level_ratio 1.1 \
  --ft_iterations 30000 --save_images
```

The checkpoint directory is the one holding `point_cloud.ply` together with the
MLP checkpoints: the decoders are global, so every level shares them.

### Key arguments

| Argument | Meaning |
|---|---|
| `--num_levels` | Number of LoDs exposed in the bitstream |
| `--prune_levels` | Internal pruning steps; the exposed levels are picked equidistantly among them |
| `--hier_min` | Number of anchors at the coarsest level, as a count or a fraction of the model |
| `--gradient_threshold` | Primitives whose normalized accumulated gradient falls below it are dropped from the finest level. This is how the finest level is chosen by default |
| `--hier_max` | Optional override: fix the finest level to a fraction of the model instead |
| `--sample_level_ratio` | Level sampling during fine-tuning: `1` uniform, `>1` favours fine levels, `<-1` coarse ones |
| `--color_l1_weight` | L1 shrinkage of the anchor features (disabled by default) |
| `--dict_size` | Size of the zstd dictionary shared by every chunk |
| `--lr_start_iteration` | Iteration the learning-rate schedules resume from; defaults to the checkpoint's iteration so fine-tuning continues the pre-training schedule |
| `--encode_eval_iters` | Iterations at which a full encode/decode/evaluate snapshot is written |

### Output

```
output/gode/bonsai/
├── masks.pt                  level masks
├── scalable_results.csv      one row per (snapshot iteration, level)
└── iter_30000/
    ├── zstd_dict.pkl         dictionary shared by every chunk
    ├── lod_0.zst ...         disjoint chunks, coarsest first
    └── renders/lod_*/        decoded test renders (--save_images)
```

`scalable_results.csv` reports, per level: PSNR, SSIM, LPIPS, number of anchors,
cumulative size in MB (dictionary included), rendering FPS, and cumulative
decoding time.

### Ablations

```shell
--prune_type gradient_oneshot|random   # importance criterion
--no_qat                               # float16 bitstream, no quantization-aware training
--pruning_only                         # hierarchy and bitstream without joint fine-tuning
```

### Bitstream

Anchor positions, offsets and scalings are stored as float16 and the anchor
features as int8 with the scale learned during quantization-aware training;
`--no_qat` stores everything as float16. Every chunk is Morton-sorted before
entropy coding so that neighbouring anchors compress together.

The neural decoders (`mlp_opacity`, `mlp_cov`, `mlp_color`, and the appearance
embedding when enabled) are global rather than per-anchor: they are shared by
every level and `Scene.save()` writes them next to the bitstream, so they are
**not** included in the per-level sizes reported in `scalable_results.csv`.
They amount to a few thousand parameters.

### Tools

```shell
# aggregate every scene into one table (per-scene rows + per-dataset averages)
python tools/make_results_table.py --run_root output/gode
```

---

## Porting GoDe to another backbone

`gode/` never touches primitives directly. A backbone is any model exposing:

| Group | Methods |
|---|---|
| Rendering | `render(camera, mask=None, measure_fps=False)` returning `{"render": ...}` |
| Optimisation | `training_setup(args)`, `update_learning_rate(it)`, `optimizer`, `train()`, `eval()`, `geometry_groups()` |
| Primitives | `num_primitives`, `primitive_names()`, `index_primitives(idx)`, `subset(mask)`, `append_primitives(other)`, `clone_empty()` |
| Importance | `_get_gradient_scores()`, `gradient_prune(k, prune_type, scores)` |
| Priors | `appearance_regularization(mask, weight)` (may return 0) |
| Compression | `_sort_morton()`, `set_qa()`, `quantizer_names()`, `calibrate_quantizers_for_compression()`, `encode_payload(quantize)`, `decode_payload(payload)` |

`encode_payload` returns `{"geom": {...}, "features": {...}}`, where arrays are
numpy and everything else is metadata; `gode/streams.py` handles dictionary
training, the container format and incremental decoding. Adding a backbone
therefore means implementing these methods on its model class and pointing a
copy of `scalable_train.py` at it — the hierarchy, the fine-tuning schedule and
the codec are unchanged.

See [`scene/gs_model_scaffoldgs/lod_model.py`](scene/gs_model_scaffoldgs/lod_model.py)
for this backbone's implementation.

---

## Acknowledgements

Built on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
and [Scaffold-GS](https://github.com/city-super/Scaffold-GS).

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
