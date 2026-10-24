"""Joint fine-tuning of all LoDs under quantization-aware training.

Every iteration samples one level and optimises the primitives it contains, so
a single set of parameters has to serve every level at once. Geometry is
frozen for the last iterations, after which only appearance adapts to the
quantizers.
"""

import time
from random import randint

import torch
from tqdm import tqdm

from gode.compression import snapshot
from gode.evaluation import report

def sampling_probabilities(num_levels, ratio, device="cpu"):
    """Level sampling distribution; |ratio| is the geometric factor between
    consecutive levels. ratio=1 is uniform, ratio>1 favours the finest level,
    ratio<-1 favours the coarsest one."""
    exponents = torch.arange(num_levels, dtype=torch.float32, device=device)
    if ratio >= 0:
        exponents = exponents.flip(0)
    weights = abs(float(ratio)) ** exponents
    return weights / weights.sum()


def parse_iterations(spec):
    if not spec or not str(spec).strip():
        return []
    return [int(x) for x in str(spec).split(",") if x.strip()]


def train(scene, model, masks, args, rows=None):
    if not model.qa and not args.no_qat:
        model.set_qa()
    model.training_setup(args)
    model.train()

    probabilities = sampling_probabilities(len(masks), args.sample_level_ratio)
    print("[Sampling] " + " ".join(
        f"L{len(masks) - 1 - i}={100.0 * p:.1f}%" for i, p in enumerate(probabilities)))
    levels = torch.multinomial(probabilities, args.ft_iterations, replacement=True)

    snapshots = {i for i in parse_iterations(args.encode_eval_iters) if 0 < i <= args.ft_iterations}
    snapshots.add(args.ft_iterations)
    print(f"[Snapshot] encode/decode at {sorted(snapshots)}")

    geometry_groups = model.geometry_groups()
    # Backbones that fine-tune a converged model continue its schedule instead
    # of restarting it; see --lr_start_iteration.
    lr_offset = getattr(args, "lr_start_iteration", 0)
    cameras = []
    ema_loss = 0.0
    frozen = False
    freeze_from = args.ft_iterations - args.freeze_geometry_iters
    progress = tqdm(range(1, args.ft_iterations + 1), desc="Scalable QA training")

    train_time = 0.0
    start = time.time()
    for iteration in range(1, args.ft_iterations + 1):
        model.update_learning_rate(lr_offset + iteration)

        if not frozen and iteration > freeze_from:
            for group in model.optimizer.param_groups:
                if group["name"] in geometry_groups:
                    group["lr"] = 0.0
            frozen = True
            print(f"[Freeze] Geometry frozen at iteration {iteration}")

        mask = masks[levels[iteration - 1].item()]
        if not cameras:
            cameras = scene.getTrainCameras().copy()
        camera = cameras.pop(randint(0, len(cameras) - 1))

        image = model.render(camera, mask=mask)["render"]
        gt = camera.original_image.cuda()
        loss = model.photometric_loss(image, gt, args.lambda_dssim)
        loss = loss + model.appearance_regularization(mask, args.appearance_reg_weight)
        loss.backward()

        ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
        if iteration % 10 == 0:
            progress.set_postfix({"Loss": f"{ema_loss:.6f}",
                                  "LoD": f"L{len(masks) - 1 - levels[iteration - 1].item()}"})
            progress.update(10)

        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)

        if args.test_every > 0 and iteration % args.test_every == 0:
            for i, level_mask in enumerate(masks):
                report(scene, model, mask=level_mask, level=args.num_levels - 1 - i)

        if iteration in snapshots:
            train_time = time.time() - start
            snapshot(scene, model, masks, args, iteration, train_time,
                     rows if rows is not None else [])
            start = time.time() - train_time  # exclude snapshot time from the budget

    progress.close()
    print(f"[Train] Fine-tuning done in {train_time:.2f}s")
    return train_time


def warmup(scene, model, args, iterations):
    """Optimise the full model with the quantizers active, so that their
    observers are calibrated before the hierarchy is built."""
    model.set_qa()
    model.training_setup(args)
    model.train()

    cameras = []
    ema_loss = 0.0
    lr_offset = getattr(args, "lr_start_iteration", 0)
    progress = tqdm(range(1, iterations + 1), desc="QAT warm-up")
    for iteration in range(1, iterations + 1):
        model.update_learning_rate(lr_offset + iteration)
        if not cameras:
            cameras = scene.getTrainCameras().copy()
        camera = cameras.pop(randint(0, len(cameras) - 1))

        image = model.render(camera)["render"]
        gt = camera.original_image.cuda()
        loss = model.photometric_loss(image, gt, args.lambda_dssim)
        loss.backward()

        ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
        if iteration % 10 == 0:
            progress.set_postfix({"Loss": f"{ema_loss:.6f}"})
            progress.update(10)

        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
    progress.close()
