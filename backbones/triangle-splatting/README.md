# GoDe: Gaussians on Demand for Backbone-Agnostic Progressive Compression

Francesco Di Sario, Riccardo Renzulli, Marco Grangetto, Akihiro Sugimoto, Enzo Tartaglione

[Project page](https://gaussians-on-demand.github.io/) · [arXiv](https://arxiv.org/abs/2501.13558)

GoDe turns a pre-trained radiance field into a **progressive, level-of-detail
bitstream**. A gradient-based importance criterion organises the primitives
into nested levels, all levels are fine-tuned jointly under quantization-aware
training, and the levels are stored as disjoint chunks sharing a trained zstd
dictionary. A client decodes chunk 0 to display the coarsest model and keeps
appending chunks to refine it, without ever re-downloading what it already has.

This folder is the **Triangle Splatting** backbone of the GoDe repository ([overview](../../README.md)). The method core is [`gode/`](../../gode), shared by all five backbones.

---

## Installation

```shell
conda create -y -n gode-triangles python=3.10
conda activate gode-triangles
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install submodules/diff-triangle-rasterization submodules/simple-knn
```

## Data

The benchmarks are expected in COLMAP layout under `$DATA_ROOT`:

```
$DATA_ROOT/nerf_real_360/{bicycle,garden,...}
$DATA_ROOT/tandt/{train,truck}
```

> MipNeRF-360 is evaluated on `images_2` for indoor scenes and `images_4` for
> outdoor ones, which `config/indoor.json` and `config/outdoor.json` already set.

## Usage

GoDe runs in two stages.

**1. Pre-train the backbone** (or bring your own checkpoint):

```shell
python train.py \
  --config config/indoor.json --scene bonsai --eval \
  -s $DATA_ROOT/nerf_real_360/bonsai -m output/bonsai
```

**2. Build and compress the hierarchy:**

```shell
python scalable_train.py \
  --config config/indoor.json --scene bonsai --eval \
  -s $DATA_ROOT/nerf_real_360/bonsai -m output/gode/bonsai \
  --checkpoint output/bonsai/point_cloud/iteration_30000 \
  --num_levels 8 --hier_min 150000 --sample_level_ratio 1.1 \
  --ft_iterations 30000 --save_images
```

The checkpoint may be an iteration directory, its `point_cloud_state_dict.pt`,
or a `.ply`.

Whole benchmarks:

```shell
DATA_ROOT=/path/to/datasets scripts/train_mipnerf360.sh      # stage 1
DATA_ROOT=/path/to/datasets scripts/scalable_mipnerf360.sh   # stage 2
```

### Key arguments

| Argument | Meaning |
|---|---|
| `--num_levels` | Number of LoDs exposed in the bitstream |
| `--prune_levels` | Internal pruning steps; the exposed levels are picked equidistantly among them |
| `--hier_min` | Size of the coarsest level, as a count or a fraction of the model |
| `--gradient_threshold` | Primitives whose normalized accumulated gradient falls below it are dropped from the finest level. This is how the finest level is chosen by default |
| `--hier_max` | Optional override: fix the finest level to a fraction of the model instead |
| `--sample_level_ratio` | Level sampling during fine-tuning: `1` uniform, `>1` favours fine levels, `<-1` coarse ones |
| `--appearance_reg_weight` | Weight of the backbone appearance prior; here an L1 sparsity on the non-DC SH coefficients |
| `--encode_eval_iters` | Iterations at which a full encode/decode/evaluate snapshot is written |

### Output

```
output/gode/bonsai/
├── masks.pt                  level masks
├── scalable_results.csv      the uncompressed baseline, then one row per (snapshot iteration, level)
└── iter_30000/
    ├── zstd_dict.pkl         dictionary shared by every chunk
    ├── lod_0.zst ...         disjoint chunks, coarsest first
    └── renders/lod_*/        decoded test renders (--save_images)
```

`scalable_results.csv` reports, per level: PSNR, SSIM, LPIPS, number of
primitives, cumulative size in MB (dictionary included), rendering FPS, and
cumulative decoding time.

### Ablations

```shell
--prune_type random|gradient_oneshot   # importance criterion
--no_qat                               # float16 bitstream, no quantization-aware training
--pruning_only                         # hierarchy and bitstream without joint fine-tuning
--baseline_eval_only                   # evaluate the pre-trained checkpoint and stop
```

---

## Porting GoDe to another backbone

`gode/` never touches primitives directly. A backbone is any model exposing:

| Group | Methods |
|---|---|
| Rendering | `render(camera, mask=None, measure_fps=False)` returning `{"render": ...}` |
| Optimisation | `training_setup(args)`, `update_learning_rate(it)`, `optimizer`, `train()`, `eval()` |
| Primitives | `num_primitives`, `primitive_names()`, `index_primitives(idx)`, `subset(mask)`, `append_primitives(other)`, `clone_empty()` |
| Importance | `_get_gradient_scores()`, `gradient_prune(k, prune_type, scores)` |
| Objective | `photometric_loss(image, gt, lambda_dssim)`, `appearance_regularization(mask, weight)`, `geometry_groups()` |
| Compression | `_sort_morton()`, `set_qa()`, `quantizer_names()`, `calibrate_quantizers_for_compression()`, `encode_payload(quantize)`, `decode_payload(payload)` |

The core is self-contained: it imports nothing from this repository, and
`pytest gode/tests` exercises the stream container and the payload contract on
synthetic data, without a GPU or a dataset. It is maintained as a single
upstream copy shared by every backbone.

`encode_payload` returns `{"geom": {...}, "features": {...}}`, where arrays are
numpy and everything else is metadata; `gode/streams.py` handles dictionary
training, the container format and incremental decoding. Adding a backbone
therefore means implementing these methods on its model class and pointing a
copy of `scalable_train.py` at it — the hierarchy, the fine-tuning schedule and
the codec are unchanged.

See [`scene/triangle_model.py`](scene/triangle_model.py) for this backbone's
implementation.

---

## Acknowledgements

Built on [Triangle Splatting](https://github.com/trianglesplatting/triangle-splatting)
and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

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
