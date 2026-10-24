#!/usr/bin/env python3
"""Build the final per-scene and per-dataset LOD summary CSV."""

import csv
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

DATASETS = {
    "mipnerf360": [
        "bicycle", "flowers", "garden", "stump", "treehill",
        "room", "counter", "kitchen", "bonsai",
    ],
    "tandt": ["train", "truck"],
    "db": ["drjohnson", "playroom"],
}
FIELDS = ["scene", "lod_level", "psnr", "ssim", "lpips", "size (in mb)", "fps"]


def final_rows(root: Path, scene: str):
    scene_dir = root / scene
    with (scene_dir / "scalable_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    final_iter = max(int(row["iter"]) for row in rows if row["level"].isdigit())
    metrics = {int(row["level"]): row for row in rows
               if int(row["iter"]) == final_iter and row["level"].isdigit()}

    lod_dir = scene_dir / f"iter_{final_iter}"
    dictionary_bytes = (lod_dir / "zstd_dict.pkl").stat().st_size
    cumulative_bytes = dictionary_bytes
    result = []
    for level in sorted(metrics):
        cumulative_bytes += (lod_dir / f"lod_{level}.zst").stat().st_size
        row = metrics[level]
        result.append({
            "scene": scene,
            "lod_level": level,
            "psnr": float(row["psnr"]),
            "ssim": float(row["ssim"]),
            "lpips": float(row["lpips"]),
            "size (in mb)": cumulative_bytes / (1024 * 1024),
            "fps": float(row["fps"]),
        })
    return result


def formatted(row):
    return [
        row["scene"],
        str(row["lod_level"]),
        f'{row["psnr"]:.6f}',
        f'{row["ssim"]:.6f}',
        f'{row["lpips"]:.6f}',
        f'{row["size (in mb)"]:.4f}',
        f'{row["fps"]:.4f}',
    ]


def main():
    parser = ArgumentParser(description="Aggregate the per-scene GoDe results into one table")
    parser.add_argument("--run_root", type=Path, required=True,
                        help="Directory holding one subdirectory per scene")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num_levels", type=int, default=8)
    args = parser.parse_args()
    output = args.output or args.run_root / "final_results.csv"

    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(FIELDS)
        for dataset, scenes in DATASETS.items():
            by_level = defaultdict(list)
            missing_scenes = []
            for scene in scenes:
                try:
                    rows = final_rows(args.run_root, scene)
                except FileNotFoundError:
                    missing_scenes.append(scene)
                    for level in range(args.num_levels):
                        writer.writerow([scene, level, "", "", "", "", ""])
                    writer.writerow([])
                    continue
                for row in rows:
                    writer.writerow(formatted(row))
                    by_level[row["lod_level"]].append(row)
                writer.writerow([])  # blank line after each scene

            for level in sorted(by_level):
                rows = by_level[level]
                average = {key: sum(row[key] for row in rows) / len(rows)
                           for key in ("psnr", "ssim", "lpips", "size (in mb)", "fps")}
                suffix = (f"_available_{len(scenes) - len(missing_scenes)}_of_{len(scenes)}"
                          if missing_scenes else "")
                average.update(scene=f"{dataset}_average{suffix}", lod_level=level)
                writer.writerow(formatted(average))
            writer.writerow([])  # an additional blank line after each dataset

    print(output)


if __name__ == "__main__":
    main()
