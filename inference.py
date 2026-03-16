import os
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageDraw
from dataclasses import dataclass
from torchvision.models.optical_flow import raft_large
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import Origin as org
import Spine as Model
from model import ConvAutoencoder

# =========================================================
# DEVICE
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# PATHS
# =========================================================
UNET_WEIGHTS = "./weights/Lightning.pth"
DECODER_WEIGHTS = "./weights/model_final.pth"
RAFT_WEIGHTS = "./weights/raft_large_offline.pth"

# =========================================================
# PROGRESS CALLBACK
# =========================================================
class ProgressCallback:
    def update(self, percent: float, message: str):
        print(f"[{percent:6.2f}%] {message}")

# =========================================================
# CONFIG
# =========================================================
@dataclass
class InferenceConfig:
    guidance_peak_scale: float = 0.005
    guidance_min_scale: float = 0.0005
    guidance_peak_step: int = 500
    guidance_width: int = 200
    guidance_interval: int = 10 
    sample_steps: int = 1000
    vis_freq: int = 25           

# =========================================================
# METRICS
# =========================================================
ssim_metric = StructuralSimilarityIndexMeasure().to(DEVICE)
ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure().to(DEVICE)

# =========================================================
# UTILS
# =========================================================
SEED = 666

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def guidance_schedule(t, cfg):
    if cfg.guidance_interval <= 0:
        return 0.0  # physics OFF
    g = math.exp(-((t - cfg.guidance_peak_step) ** 2) / (2 * cfg.guidance_width ** 2))
    return max(g * cfg.guidance_peak_scale, cfg.guidance_min_scale)

def load_png_gray(path):
    img = Image.open(path).convert("L")
    tfm = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    img = tfm(img).unsqueeze(0).to(DEVICE)
    return img * 2 - 1

def load_lpips():
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex",
        normalize=True
    ).to(DEVICE)
    lpips.eval()
    return lpips

def compute_metrics(pred, gt, lpips_metric):
    p = torch.clamp((pred + 1) / 2, 0, 1)
    g = torch.clamp((gt + 1) / 2, 0, 1)

    mse = F.mse_loss(p, g).item()
    mae = F.l1_loss(p, g).item()
    ssim = ssim_metric(p, g).item()
    ms_ssim = ms_ssim_metric(p, g).item()

    p3 = p.repeat(1, 3, 1, 1)
    g3 = g.repeat(1, 3, 1, 1)
    lpips = lpips_metric(p3, g3).item()

    return mse, mae, ssim, ms_ssim, lpips

# =========================================================
# PHYSICS
# =========================================================
class AdvectionSolver(torch.nn.Module):
    def __init__(self, H=512, W=512):
        super().__init__()
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing="ij"
        )
        self.register_buffer("grid", torch.stack([x, y], dim=-1))
        self.H, self.W = H, W

    def warp(self, img, flow):
        flow = flow.permute(0, 2, 3, 1)
        flow[..., 0] *= 2 / (self.W - 1)
        flow[..., 1] *= 2 / (self.H - 1)
        return F.grid_sample(img, self.grid - flow, align_corners=True, padding_mode="border")

    def forward(self, pred, prev, flow, mask):
        target = self.warp(prev, flow).detach()
        loss = torch.mean(mask * torch.abs(pred - target))
        return loss, target

# =========================================================
# MODELS
# =========================================================
def load_models(progress=None):
    if progress: progress.update(2, "Loading UNet & Autoencoder")

    unet = Model.UNet().to(DEVICE).eval()
    unet.load_state_dict(torch.load(UNET_WEIGHTS, map_location=DEVICE))

    cae = ConvAutoencoder(1, 64, 4).to(DEVICE).eval()
    cae.load_state_dict(torch.load(DECODER_WEIGHTS, map_location=DEVICE))

    if progress: progress.update(6, "Loading RAFT")
    raft = raft_large(weights=None).to(DEVICE).eval()
    raft.load_state_dict(torch.load(RAFT_WEIGHTS, map_location=DEVICE))

    return unet, cae, raft

# =========================================================
# SAMPLING (FIXED)
# =========================================================
def run_sampling(unet, cae, context, past_imgs, flow, mask, solver, cfg, progress=None, image_callback=None):
    betas = Model.cosine_beta_schedule(org.diff_timesteps).to(DEVICE)
    alphas = 1 - betas
    ab = torch.cumprod(alphas, dim=0)

    x = torch.randn((1, 4, 64, 64), device=DEVICE)
    grad_buf = torch.zeros_like(x)

    total = cfg.sample_steps
    
    # We now use cfg.vis_freq directly inside the loop

    for step_idx, i in enumerate(reversed(range(cfg.sample_steps))):
        scale = guidance_schedule(i, cfg)

        if cfg.guidance_interval > 0 and scale > 0 and i % cfg.guidance_interval == 0:
            x_in = x.detach().requires_grad_(True)
            eps = unet(x_in, torch.tensor([i], device=DEVICE), context)
            x0 = (x_in - torch.sqrt(1 - ab[i]) * eps) / torch.sqrt(ab[i])
            img0 = cae.decode(x0)

            loss, _ = solver(img0, past_imgs[-1], flow, mask)
            grad = torch.autograd.grad(loss, x_in)[0]
            grad_buf = grad / (grad.std() + 1e-8) * scale
        else:
            grad_buf.zero_()

        with torch.no_grad():
            eps = unet(x, torch.tensor([i], device=DEVICE), context)
            x = (1 / torch.sqrt(alphas[i])) * (x - betas[i] / torch.sqrt(1 - ab[i]) * eps) - grad_buf

            # UPDATED: Uses cfg.vis_freq to control callback frequency
            if image_callback and step_idx % cfg.vis_freq == 0:
                image_callback(cae.decode(x))

        if progress and step_idx % 10 == 0:
            progress.update(40 + 50 * step_idx / total, f"Sampling {step_idx}/{total}")

    return cae.decode(x)

# =========================================================
# CORE PIPELINE
# =========================================================
def _run_core(unet, cae, raft, imgs, cfg, progress=None, image_callback=None, lpips_metric=None):
    past_imgs, gt_img = imgs[:4], imgs[4]

    with torch.no_grad():
        latents = [cae.encode(im) for im in past_imgs]
        context = torch.cat(latents, dim=1)

        to_rgb = lambda x: x.repeat(1, 3, 1, 1).add(1).div(2)
        flow = raft(to_rgb(past_imgs[-2]), to_rgb(past_imgs[-1]))[-1]

        solver = AdvectionSolver().to(DEVICE)
        warped = solver.warp(past_imgs[-1], flow)
        err = torch.abs(warped - past_imgs[-1])
        mask = torch.clamp(torch.exp(-err / (err.mean() + 1e-6)), 0.05, 1.0)

    pred = run_sampling(unet, cae, context, past_imgs, flow, mask, solver, cfg, progress, image_callback)
    metrics = compute_metrics(pred, gt_img, lpips_metric)
    diff_map = torch.abs(pred - gt_img).clamp(0, 1)

    return past_imgs, pred, gt_img, metrics, diff_map

# =========================================================
# PUBLIC API
# =========================================================
def run_inference_from_images(cfg, image_paths, progress=None, image_callback=None):
    set_seed(SEED)
    unet, cae, raft = load_models(progress)
    lpips_metric = load_lpips()
    imgs = [load_png_gray(p) for p in image_paths]
    return _run_core(unet, cae, raft, imgs, cfg, progress, image_callback, lpips_metric)