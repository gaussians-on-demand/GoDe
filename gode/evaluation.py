"""Per-LoD evaluation and result bookkeeping."""

import csv
import os

import torch

from gode.metrics import lpips_model, psnr, ssim

CSV_FIELDS = [
    "iter", "level", "psnr", "ssim", "lpips", "num_primitives",
    "cumulative_size_mb", "fps", "dec_time_s", "enc_time_s",
    "train_time_s", "iters",
]

@torch.no_grad()
def evaluate(scene, model, mask=None, eval_lpips=True, save_dir=None):
    """Render the test split; returns (psnr, ssim, lpips, fps, num_primitives)."""
    model.eval()
    cameras = scene.getTestCameras()
    device = torch.device("cuda")
    lpips = lpips_model(device) if eval_lpips else None

    if save_dir is not None:
        from PIL import Image
        os.makedirs(save_dir, exist_ok=True)

    psnr_acc = ssim_acc = lpips_acc = fps_acc = 0.0
    for idx, viewpoint in enumerate(cameras):
        pkg = model.render(viewpoint, mask=mask, measure_fps=True)
        image = torch.clamp(pkg["render"], 0.0, 1.0)
        gt = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)
        psnr_acc += psnr(image, gt).mean().item()
        ssim_acc += ssim(image.unsqueeze(0), gt.unsqueeze(0)).item()
        fps_acc += pkg.get("fps", 0.0)
        if eval_lpips:
            lpips_acc += lpips(image * 2 - 1, gt * 2 - 1).mean().item()
        if save_dir is not None:
            name = getattr(viewpoint, "image_name", f"frame_{idx:04d}")
            array = (image.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
            Image.fromarray(array).save(os.path.join(save_dir, f"{name}.png"))

    n = max(len(cameras), 1)
    num_primitives = int(mask.sum().item()) if mask is not None else model.num_primitives
    return psnr_acc / n, ssim_acc / n, lpips_acc / n, fps_acc / n, num_primitives


@torch.no_grad()
def report(scene, model, mask=None, level=None):
    """Quick evaluation printed during training; never written to the CSV."""
    quantizers = capture_quantizers(model)
    try:
        psnr_v, ssim_v, _, fps_v, num = evaluate(scene, model, mask=mask, eval_lpips=False)
        tag = f"L{level}" if level is not None else "full"
        print(f"[{tag}] PSNR={psnr_v:.2f} SSIM={ssim_v:.3f} FPS={fps_v:.1f} N={num}")
    finally:
        restore_quantizers(model, quantizers)
        model.train()


def capture_quantizers(model):
    """Snapshot the fake-quantize observers, which evaluation would otherwise update."""
    if not getattr(model, "qa", False):
        return None
    import copy
    return {name: copy.deepcopy(getattr(model, name).state_dict())
            for name in model.quantizer_names()}


def restore_quantizers(model, state):
    if state:
        for name, values in state.items():
            getattr(model, name).load_state_dict(values)


@torch.no_grad()
def baseline_row(scene, model, iteration=0):
    """Reference row for the uncompressed checkpoint the hierarchy starts from."""
    psnr_v, ssim_v, lpips_v, fps_v, num = evaluate(scene, model)
    model.train()
    print(f"[Baseline] PSNR={psnr_v:.4f} SSIM={ssim_v:.6f} LPIPS={lpips_v:.6f} "
          f"FPS={fps_v:.2f} N={num}")
    return {"iter": iteration, "level": "baseline", "psnr": psnr_v, "ssim": ssim_v,
            "lpips": lpips_v, "num_primitives": num, "cumulative_size_mb": "",
            "fps": fps_v, "dec_time_s": "", "enc_time_s": "", "train_time_s": 0.0,
            "iters": 0}


def write_csv(model_path, rows):
    path = os.path.join(model_path, "scalable_results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path
