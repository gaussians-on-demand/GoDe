"""Gradient-based construction of the LoD hierarchy.

Level 0 is the coarsest level (fewest primitives) and level L-1 the finest.
Internally the masks are built from finest to coarsest: the finest level is
obtained by physically dropping the primitives whose accumulated gradient is
below `gradient_threshold`, and every further level prunes the previous one,
accumulating gradients through the mask of the level above so that importance
reflects the configuration that is actually rendered.
"""

import time

import numpy as np
import torch
from tqdm import tqdm

from gode.evaluation import report


def geometric_progression(start, stop, steps):
    ratio = np.exp((np.log(stop) - np.log(start)) / (steps - 1))
    return ((ratio ** np.arange(steps)) * start).astype(np.int64)


def resolve_minimum(value, num_primitives):
    """--hier_min accepts an absolute count or a fraction of the loaded model."""
    if value is None:
        return max(1, int(0.10 * num_primitives))
    value = float(value)
    if 0 < value < 1:
        return max(1, int(value * num_primitives))
    if value >= 1:
        return int(value)
    raise ValueError("--hier_min must be a positive count or a fraction in (0, 1)")


def accumulate_gradients(scene, model, args, mask=None):
    """Backward pass over every training camera, optionally through a LoD mask."""
    model.train()
    for camera in tqdm(scene.getTrainCameras(), desc="Accumulating gradients", leave=False):
        image = model.render(camera, mask=mask)["render"]
        gt = camera.original_image.cuda()
        loss = model.photometric_loss(image, gt, args.lambda_dssim)
        loss.backward(retain_graph=True)


def build(scene, model, args):
    """Returns (masks, elapsed_seconds); masks[0] is the finest level."""
    start = time.time()
    num_primitives = model.num_primitives
    # Pruning granularity is decoupled from the number of exposed levels.
    steps = max(args.prune_levels, args.num_levels)

    finest_mask = None
    oneshot_scores = None

    if args.hier_max is None:
        accumulate_gradients(scene, model, args)
        scores = model._get_gradient_scores()
        finest_mask = scores >= args.gradient_threshold
        maximum = int(finest_mask.sum().item())
        if maximum == 0:
            raise RuntimeError(
                "The gradient threshold removed every primitive; lower "
                "--gradient_threshold or pass --hier_max explicitly."
            )
        print(f"[Hierarchy] Finest level from gradient threshold "
              f">={args.gradient_threshold:g}: {maximum}/{num_primitives} "
              f"({maximum / num_primitives:.2%})")
        if args.prune_type == "gradient_oneshot":
            oneshot_scores = scores
        model.optimizer.zero_grad(set_to_none=True)
    else:
        maximum = int(num_primitives * args.hier_max)
        if args.prune_type == "gradient_oneshot":
            accumulate_gradients(scene, model, args)
            oneshot_scores = model._get_gradient_scores()
            model.optimizer.zero_grad(set_to_none=True)

    minimum = resolve_minimum(args.hier_min, num_primitives)
    if not 0 < minimum <= maximum:
        raise ValueError(f"hier_min must be in (0, {maximum}], got {minimum}.")

    # Number of primitives to drop at each step, finest level first.
    to_prune = list(np.flip(num_primitives - geometric_progression(minimum, maximum, steps)))
    print(f"[Hierarchy] N={num_primitives} steps={steps} levels={args.num_levels} "
          f"prune={to_prune}")

    masks = []
    active = None
    for step in range(steps):
        if step == 0 and finest_mask is not None:
            mask = finest_mask
        else:
            if oneshot_scores is None and args.prune_type != "random":
                accumulate_gradients(scene, model, args, mask=active)
            k = int(to_prune[step])
            if step > 0:
                k -= int(to_prune[0])
            mask, _ = model.gradient_prune(k, prune_type=args.prune_type, scores=oneshot_scores)

        if step == 0:
            # Drop the primitives outside the finest level for good, then
            # restart the optimizer on the smaller model.
            with torch.no_grad():
                model.index_primitives(mask)
                if oneshot_scores is not None:
                    oneshot_scores = oneshot_scores[mask.cpu()]
                mask = torch.ones(model.num_primitives, dtype=torch.bool, device="cuda")
                model.training_setup(args)

        active = mask
        masks.append(mask.cuda())
        model.optimizer.zero_grad(set_to_none=True)
        report(scene, model, mask=mask, level=steps - 1 - step)

    selected = np.round(np.linspace(0, steps - 1, args.num_levels)).astype(int)
    print(f"[Hierarchy] Selected steps {selected.tolist()} as the {args.num_levels} LoDs")
    elapsed = time.time() - start
    print(f"[Hierarchy] Built in {elapsed:.2f}s")
    return [masks[i] for i in selected], elapsed
