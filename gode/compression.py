"""Progressive encoding and incremental decoding of the LoD hierarchy.

The levels are nested, so only their differences are stored: the coarsest
level is written in full and every finer level adds the primitives it does not
share with the previous one. A client therefore reaches level L by decoding
chunks 0..L, and the size reported for a level is the cumulative size of those
chunks plus the shared zstd dictionary.
"""

import os
import time

import torch

from gode import streams
from gode.evaluation import capture_quantizers, evaluate, report, restore_quantizers, write_csv


def disjoint_chunks(masks):
    """Nested level masks (finest first) -> disjoint chunks (coarsest first)."""
    chunks = [masks[-1]]
    for i in range(len(masks) - 1, 0, -1):
        chunks.append(torch.logical_xor(masks[i - 1], masks[i]))
    return chunks


def encode(model, masks, out_dir, quantize=True, dict_size=262_144):
    """Write one .zst per chunk; returns (chunks, dict_path, sizes_mb, times_s)."""
    with torch.no_grad():
        if quantize:
            model.calibrate_quantizers_for_compression()
        else:
            print("[ENC] No-QAT mode: float16 payloads, no shared dictionary")

        chunks = disjoint_chunks([m.cuda() for m in masks])
        dict_path = None
        dicts = None
        if quantize:
            dict_path = os.path.join(out_dir, "zstd_dict.pkl")
            t0 = time.time()
            dicts = streams.train_dictionaries(model.encode_payload(), dict_size=dict_size,
                                               save_path=dict_path)
            print(f"[ENC] Dictionary trained in {time.time() - t0:.3f}s")

        sizes, times = [], []
        for i, chunk in enumerate(chunks):
            part = model.subset(chunk)
            part._sort_morton()
            t0 = time.time()
            size = streams.write(os.path.join(out_dir, f"lod_{i}.zst"),
                                 part.encode_payload(quantize=quantize), dicts)
            times.append(time.time() - t0)
            sizes.append(size)
            print(f"[ENC] lod_{i}: N={int(chunk.sum())} {size:.2f} MB in {times[-1]:.3f}s")

    return chunks, dict_path, sizes, times


def decode_chunk(model, path, dicts=None):
    """Decode one .zst chunk into a fresh model of the same kind."""
    chunk = model.clone_empty()
    chunk.decode_payload(streams.read(path, dicts))
    return chunk


def decode_levels(model, out_dir, level, dict_path=None):
    """Decode chunks 0..level and merge them into a single renderable model."""
    dicts = streams.load_dictionaries(dict_path)
    accumulated = decode_chunk(model, os.path.join(out_dir, "lod_0.zst"), dicts)
    for i in range(1, level + 1):
        accumulated.append_primitives(decode_chunk(model, os.path.join(out_dir, f"lod_{i}.zst"), dicts))
    return accumulated


def decode_all_levels(model, out_dir, dict_path=None, max_level=None):
    """Decode every level once, returning one renderable model per level.

    Level k holds chunks 0..k, so the returned models are the operating points
    a client reaches as the stream arrives.
    """
    import copy

    dicts = streams.load_dictionaries(dict_path)
    level = 0
    accumulated, models = None, []
    while os.path.exists(os.path.join(out_dir, f"lod_{level}.zst")):
        if max_level is not None and level > max_level:
            break
        chunk = decode_chunk(model, os.path.join(out_dir, f"lod_{level}.zst"), dicts)
        accumulated = chunk if accumulated is None else accumulated
        if level > 0:
            accumulated.append_primitives(chunk)
        models.append(copy.deepcopy(accumulated))
        level += 1
    if not models:
        raise FileNotFoundError(f"No lod_*.zst found in {out_dir}")
    return models


def decode_and_evaluate(scene, model, num_chunks, dict_path, out_dir, save_dir=None):
    """Decode the chunks one by one, evaluating the scene at every level."""
    dicts = streams.load_dictionaries(dict_path)
    accumulated = None
    decode_time = 0.0
    rows = []

    for level in range(num_chunks):
        torch.cuda.synchronize()
        t0 = time.time()
        chunk = decode_chunk(model, os.path.join(out_dir, f"lod_{level}.zst"), dicts)
        if level == 0:
            accumulated = chunk
        else:
            accumulated.append_primitives(chunk)
        torch.cuda.synchronize()
        decode_time += time.time() - t0

        level_dir = os.path.join(save_dir, f"lod_{level}") if save_dir else None
        psnr_v, ssim_v, lpips_v, fps_v, num = evaluate(scene, accumulated, save_dir=level_dir)
        print(f"[DEC] L{level} dec={decode_time:.3f}s N={num} PSNR={psnr_v:.2f}")
        rows.append({"level": level, "psnr": psnr_v, "ssim": ssim_v, "lpips": lpips_v,
                     "num_primitives": num, "fps": fps_v, "dec_time_s": decode_time})
        accumulated.train()

    return rows


def snapshot(scene, model, masks, args, iteration, train_time_s, rows):
    """Encode, decode and evaluate the current model; appends to the results CSV."""
    out_dir = os.path.join(args.model_path, f"iter_{int(iteration)}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[SNAPSHOT] iter={iteration} -> {out_dir}")
    quantizers = capture_quantizers(model)

    try:
        with torch.no_grad():
            for i, mask in enumerate(masks):
                report(scene, model, mask=mask, level=args.num_levels - 1 - i)

        quantize = not args.no_qat
        chunks, dict_path, sizes, enc_times = encode(model, masks, out_dir, quantize=quantize,
                                                     dict_size=args.dict_size)
        final = args.pruning_only or int(iteration) == args.ft_iterations
        save_dir = os.path.join(out_dir, "renders") if (args.save_images and final) else None
        level_rows = decode_and_evaluate(scene, model, len(chunks), dict_path, out_dir, save_dir)

        cumulative = os.path.getsize(dict_path) / (1024 ** 2) if dict_path else 0.0
        for i, row in enumerate(level_rows):
            cumulative += sizes[i]
            row.update(iter=int(iteration), cumulative_size_mb=round(cumulative, 4),
                       enc_time_s=round(enc_times[i], 4),
                       train_time_s=round(train_time_s, 2), iters=args.ft_iterations)

        rows.extend(level_rows)
        print(f"[SNAPSHOT] results appended to {write_csv(args.model_path, rows)}")
        return level_rows
    finally:
        restore_quantizers(model, quantizers)
        model.train()
