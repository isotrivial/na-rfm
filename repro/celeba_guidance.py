#!/usr/bin/env python3
"""CelebA-HQ Wiener, RFM, and NA-RFM sampling and evaluation.

Creates a consolidated evaluation comparing three guidance approaches on CelebA:
1. Wiener-only: Gaussian filtering at x0 level (high sigma)
2. RFM-only: Activation steering via hooks (all sigma)
3. Hybrid: Wiener (high sigma) + RFM (low sigma)

Metrics computed:
- Accuracy: Attribute classification (gender, age, hair)
- KID: Kernel Inception Distance (quality metric)
- Inception Score: Image diversity/quality metric

Test combinations (8 total):
- Gender + Hair: male_blond, male_nonblond, female_blond, female_nonblond
- Age + Gender: young_male, young_female, old_male, old_female

Usage:
    cd <repo-root>
    PYTHONPATH=".:third_party/edm:baselines/tfg:$PYTHONPATH" \
      python repro/celeba_guidance.py

    # Quick test (16 samples, 1 combo)
    PYTHONPATH=".:third_party/edm:baselines/tfg:$PYTHONPATH" \
      python repro/celeba_guidance.py --quick

    # Generate baseline (no guidance)
    PYTHONPATH=".:third_party/edm:baselines/tfg:$PYTHONPATH" \
      python repro/celeba_guidance.py --baseline-only
"""

import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Optional, List, Tuple, Dict
import os

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import pipeline
from torchvision import transforms
from torchmetrics.image.kid import KernelInceptionDistance
from scipy.stats import entropy
from torchvision.models.inception import inception_v3, Inception_V3_Weights
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pca.gaussian_denoiser import GaussianDenoiser, load_gaussian_filter
from na_rfm.hooks import ActivationGuidanceHook, GuidanceWindow
from na_rfm.ddim_sampling import (
    build_ddim_schedule_from_scheduler,
    DDIMSchedule,
    predict_eps,
    ddim_step,
)

from na_rfm.celeba_config import (
    CONCEPTS_DIR,
    BLOCK_NAME,
    TIMESTEP,
)


# Paths
RESULTS_DIR = PROJECT_ROOT / "results" / "celeba" / "guidance"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WIENER_FILTER_DIR = PROJECT_ROOT / "repro" / "assets" / "celeba_filters"
CELEBA_DATA_DIR = PROJECT_ROOT / "data" / "celeba_hq_256_30k"

CELEBA_DDPM_MODEL_ID = "google/ddpm-ema-celebahq-256"

# Evaluation classifiers
EVAL_GENDER_MODEL = "rizvandwiki/gender-classification"
EVAL_AGE_MODEL = "ibombonato/swin-age-classifier"
EVAL_HAIR_MODEL = "londe33/hair_v02"


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class WienerConfig:
    """Configuration for Wiener-only guidance."""
    wiener_strength: float = 2.0
    sigma_end: float = 3.5  # Stop guidance at this sigma
    num_steps: int = 50
    eta: float = 0.0


@dataclass
class RFMConfig:
    """Configuration for RFM-only guidance."""
    rfm_strength: float = 0.3  # Per-attribute tuned
    rfm_cfg_scale: float = 2.0
    rfm_sigma_end: float = 0.5  # Stop RFM when sigma < this (improves KID)
    topk: int = 5
    num_steps: int = 100
    eta: float = 0.0


@dataclass
class HybridConfig:
    """Configuration for hybrid Wiener + RFM guidance."""
    # Wiener phase (high sigma)
    wiener_strength: float = 2.0
    wiener_sigma_end: float = 3.5

    # RFM phase (low sigma)
    rfm_strength: float = 0.4
    rfm_cfg_scale: float = 2.0
    rfm_sigma_start: float = 3.5
    rfm_sigma_end: float = 0.5  # Stop RFM when sigma < this (improves KID)
    topk: int = 5

    # Sampling
    num_steps: int = 100
    eta: float = 0.0


# Per-attribute strength multipliers for RFM
RFM_STRENGTH_MULT = {
    "male": 1.0,
    "female": 0.5,
    "young": 0.8,
    "old": 1.0,
    "blond": 1.0,
    "nonblond": 1.0,
}

# Per-attribute strength multipliers for Wiener (like RFM_STRENGTH_MULT)
WIENER_STRENGTH_MULT = {
    "male": 1.0,
    "female": 1.0,
    "young": 1.0,
    "old": 1.0,
    "blond": 1.0,
    "nonblond": 1.0,  # Actually uses blond filter with negative strength
}

# Wiener filters
WIENER_FILTERS = {
    "male": "celeba_male.pt",
    "female": "celeba_female.pt",
    "young": "celeba_young.pt",
    "old": "celeba_old.pt",
    "blond": "celeba_blond_hair.pt",
}

# RFM concepts - (concept_name, sign)
# nonblond = blond_vs_other with sign=-1
RFM_CONCEPTS = {
    "male": ("male_vs_female", +1),
    "female": ("male_vs_female", -1),
    "young": ("young_vs_other", +1),
    "old": ("old_vs_other", +1),
    "blond": ("blond_vs_other", +1),
    "nonblond": ("blond_vs_other", -1),
}

# Test combinations
COMBOS = {
    # Gender + Hair
    "male_blond": ["male", "blond"],
    "male_nonblond": ["male", "nonblond"],
    "female_blond": ["female", "blond"],
    "female_nonblond": ["female", "nonblond"],
    # Age + Gender
    "young_male": ["young", "male"],
    "young_female": ["young", "female"],
    "old_male": ["old", "male"],
    "old_female": ["old", "female"],
}


# =============================================================================
# Model Loading
# =============================================================================

def load_celeba_model(device: str = "cuda"):
    """Load CelebA DDPM model."""
    from diffusers import DDIMPipeline
    print(f"Loading CelebA DDPM model: {CELEBA_DDPM_MODEL_ID}")
    pipe = DDIMPipeline.from_pretrained(CELEBA_DDPM_MODEL_ID)
    pipe.unet.to(device).eval()
    return pipe.unet, pipe.scheduler


def load_concept_direction(
    path: Path,
    topk: int = 5,
    device: str = "cuda",
    orthogonalize_against: torch.Tensor = None
):
    """Load concept and compute weighted direction from top-k eigenvectors."""
    data = torch.load(path, weights_only=False)
    eigenvectors = data["eigenvectors"].to(device)
    eigenvalues = data["eigenvalues"].to(device)
    sign = data["sign"]

    k = min(topk, len(eigenvalues))
    weights = eigenvalues[:k] / (eigenvalues[:k].sum() + 1e-12)
    direction = (eigenvectors[:, :k] * weights.unsqueeze(0)).sum(dim=1)
    direction = direction / (torch.norm(direction) + 1e-8)
    direction = direction * sign

    if orthogonalize_against is not None:
        proj = (direction @ orthogonalize_against) / (orthogonalize_against.norm()**2 + 1e-8)
        direction = direction - proj * orthogonalize_against
        direction = direction / (torch.norm(direction) + 1e-8)

    return direction, data.get("auc", None)


# =============================================================================
# Quality Metrics
# =============================================================================

def load_celeba_reference_images(num_samples: int = 1000) -> List[Image.Image]:
    """Load reference images from CelebA-HQ for KID computation."""
    images = []
    img_files = sorted(os.listdir(CELEBA_DATA_DIR))[:num_samples]
    for img_file in img_files:
        img_path = CELEBA_DATA_DIR / img_file
        images.append(Image.open(img_path).convert("RGB"))
    return images


@torch.no_grad()
def compute_kid(generated_images: List[Image.Image], device: str = "cuda") -> float:
    """Compute Kernel Inception Distance against CelebA-HQ reference images."""
    kid = KernelInceptionDistance(subset_size=min(len(generated_images), 50), normalize=True).to(device)

    def image_proc(images):
        tf = transforms.Compose([transforms.Resize(299), transforms.ToTensor()])
        return torch.stack([tf(img) for img in images], dim=0).to(device)

    # Load reference images
    ref_images = load_celeba_reference_images(num_samples=1000)

    batch_size = 32
    for i in range(0, len(ref_images), batch_size):
        ref_batch = ref_images[i:i+batch_size]
        ref_batch = image_proc(ref_batch)
        kid.update(ref_batch, real=True)

    for i in range(0, len(generated_images), batch_size):
        batch = generated_images[i:i+batch_size]
        batch = image_proc(batch)
        kid.update(batch, real=False)

    score = kid.compute()
    return score[0].item()


@torch.no_grad()
def compute_inception_score(images: List[Image.Image], device: str = "cuda", batch_size: int = 32) -> float:
    """Compute Inception Score for generated images."""
    N = len(images)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear')

    def get_pred(x):
        x = up(x)
        x = inception_model(x)
        return torch.nn.functional.softmax(x, dim=-1).data.cpu().numpy()

    preds = []
    for i in range(0, N, batch_size):
        batch = images[i:i+batch_size]
        batch_tensor = torch.stack([tf(img.convert('RGB')) for img in batch], dim=0).to(device)
        preds.append(get_pred(batch_tensor))

    preds = np.concatenate(preds, 0)
    py = np.mean(preds, axis=0)
    scores = []
    for i in range(preds.shape[0]):
        pyx = preds[i, :]
        scores.append(entropy(pyx, py))

    return float(np.exp(np.mean(scores)))


def compute_quality_metrics(
    images: List[Image.Image],
    device: str = "cuda",
    skip: bool = False,
) -> tuple[float, float]:
    """Return KID and Inception Score, or deterministic placeholders when skipped."""
    if skip:
        return 1.0, 0.0
    return compute_kid(images, device), compute_inception_score(images, device)


# =============================================================================
# Attribute Evaluation
# =============================================================================

def load_evaluators(device: str = "cuda") -> dict:
    """Load evaluation classifiers."""
    evaluators = {}

    print(f"Loading gender evaluator: {EVAL_GENDER_MODEL}")
    evaluators["gender"] = pipeline(
        "image-classification",
        model=EVAL_GENDER_MODEL,
        device=0 if device == "cuda" else -1,
    )

    print(f"Loading age evaluator: {EVAL_AGE_MODEL}")
    evaluators["age"] = pipeline(
        "image-classification",
        model=EVAL_AGE_MODEL,
        device=0 if device == "cuda" else -1,
    )

    print(f"Loading hair evaluator: {EVAL_HAIR_MODEL}")
    evaluators["hair"] = pipeline(
        "image-classification",
        model=EVAL_HAIR_MODEL,
        device=0 if device == "cuda" else -1,
    )

    return evaluators


def evaluate_gender(evaluator, images: list, target: str) -> float:
    """Evaluate gender accuracy."""
    results = evaluator(images, top_k=2)
    correct = 0
    for result in results:
        top_label = result[0]["label"].lower()
        if target == "female":
            if "female" in top_label or "woman" in top_label:
                correct += 1
        else:  # male
            if "male" in top_label or "man" in top_label:
                if "female" not in top_label and "woman" not in top_label:
                    correct += 1
    return correct / len(images)


def evaluate_age(evaluator, images: list, target: str) -> float:
    """Evaluate age accuracy using TFG's mean probability per age group."""
    results = evaluator(images, top_k=10)
    correct = 0
    for result in results:
        scores = {pred["label"]: pred["score"] for pred in result}
        probs = [scores[evaluator.model.config.id2label[i]] for i in range(10)]
        prediction = np.argmax([np.mean(probs[:3]), np.mean(probs[5:])])
        if prediction == (0 if target == "young" else 1):
            correct += 1

    return correct / len(images)


def evaluate_hair(evaluator, images: list, target: str) -> float:
    """Evaluate hair color accuracy, mapping nonblond to TFG's black-hair target."""
    results = evaluator(images, top_k=5)
    correct = 0
    for result in results:
        top_label = result[0]["label"].lower()
        if target == "blond":
            if "blond" in top_label or "blonde" in top_label or "yellow" in top_label:
                correct += 1
        elif target == "nonblond":
            if "black" in top_label:
                correct += 1
    return correct / len(images)


def evaluate_combo(evaluators: dict, images: list, combo_name: str) -> Dict[str, float]:
    """Evaluate all attributes for a combination."""
    results = {}

    # Gender
    if "male" in combo_name and "female" not in combo_name:
        results["gender"] = evaluate_gender(evaluators["gender"], images, "male")
    elif "female" in combo_name:
        results["gender"] = evaluate_gender(evaluators["gender"], images, "female")

    # Age
    if "young" in combo_name:
        results["age"] = evaluate_age(evaluators["age"], images, "young")
    elif "old" in combo_name:
        results["age"] = evaluate_age(evaluators["age"], images, "old")

    # Hair
    if "blond" in combo_name and "nonblond" not in combo_name:
        results["hair"] = evaluate_hair(evaluators["hair"], images, "blond")
    elif "nonblond" in combo_name:
        results["hair"] = evaluate_hair(evaluators["hair"], images, "nonblond")

    # Combined accuracy (mean of individual accuracies, as in TFG)
    if len(results) >= 2:
        results["combined"] = float(np.mean(list(results.values())))

    return results


# =============================================================================
# Image Utilities
# =============================================================================

def make_image_grid(images: torch.Tensor, nrow: int = 8) -> Image.Image:
    """Convert tensor batch to image grid."""
    images = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    B, H, W, C = images.shape
    ncol = (B + nrow - 1) // nrow
    grid = np.zeros((ncol * H, nrow * W, C), dtype=np.uint8)
    for idx, img in enumerate(images):
        i, j = divmod(idx, nrow)
        grid[i * H:(i + 1) * H, j * W:(j + 1) * W] = img
    return Image.fromarray(grid)


def tensor_to_pil(samples: torch.Tensor) -> list:
    """Convert tensor samples to PIL images."""
    samples = ((samples + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    samples = samples.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(img) for img in samples]


# =============================================================================
# Sampling Functions
# =============================================================================

@torch.no_grad()
def sample_baseline(
    model,
    scheduler,
    num_samples: int = 256,
    num_steps: int = 50,
    batch_size: int = 8,
    seed: int = 42,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate baseline samples without any guidance."""
    all_images = []
    generator = torch.Generator(device=device).manual_seed(seed)
    schedule = build_ddim_schedule_from_scheduler(scheduler, num_steps, device)

    num_batches = (num_samples + batch_size - 1) // batch_size

    for _ in tqdm(range(num_batches), desc="Baseline sampling"):
        current_batch = min(batch_size, num_samples - len(all_images))

        x = torch.randn(
            (current_batch, 3, 256, 256),
            device=device,
            generator=generator,
        )

        for step_idx in range(len(schedule.timesteps)):
            t = schedule.timesteps[step_idx]
            alpha_prod_t = schedule.alpha_prod_t[step_idx]
            alpha_prod_t_prev = schedule.alpha_prod_t_prev[step_idx]

            t_batch = torch.full((x.shape[0],), int(t), device=device, dtype=torch.long)
            eps = predict_eps(model, x, t_batch)
            x = ddim_step(x, eps, alpha_prod_t, alpha_prod_t_prev, eta=0.0, clip_x0=True)

        all_images.append(x)

    return torch.cat(all_images, dim=0)[:num_samples]


@torch.no_grad()
def sample_wiener_only(
    model,
    scheduler,
    wiener_denoisers: List[Tuple[GaussianDenoiser, float]],
    full_denoiser: GaussianDenoiser,
    config: WienerConfig,
    num_samples: int = 256,
    batch_size: int = 8,
    seed: int = 42,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate samples with Wiener-only guidance."""
    all_images = []
    generator = torch.Generator(device=device).manual_seed(seed)
    schedule = build_ddim_schedule_from_scheduler(scheduler, config.num_steps, device)

    num_batches = (num_samples + batch_size - 1) // batch_size

    for _ in tqdm(range(num_batches), desc="Wiener-only sampling"):
        current_batch = min(batch_size, num_samples - len(all_images))

        x = torch.randn(
            (current_batch, 3, 256, 256),
            device=device,
            generator=generator,
        )

        for step_idx in range(len(schedule.timesteps)):
            t = schedule.timesteps[step_idx]
            alpha_prod_t = schedule.alpha_prod_t[step_idx]
            alpha_prod_t_prev = schedule.alpha_prod_t_prev[step_idx]
            sigma = schedule.sigmas[step_idx].item()
            ab_t = float(alpha_prod_t.item())

            t_batch = torch.full((x.shape[0],), int(t), device=device, dtype=torch.long)
            eps = predict_eps(model, x, t_batch)

            # Compute Wiener guidance at x0 level
            x0_guidance = None
            if sigma >= config.sigma_end:
                x_edm = x / np.sqrt(ab_t)
                d_full = full_denoiser.denoise(x_edm.float(), sigma)

                x0_guidance = torch.zeros_like(x)
                for denoiser, strength in wiener_denoisers:
                    d_class = denoiser.denoise(x_edm.float(), sigma)
                    x0_guidance += strength * (d_class - d_full)
                x0_guidance = x0_guidance.to(x.dtype)

            x = ddim_step(
                x, eps, alpha_prod_t, alpha_prod_t_prev, config.eta,
                clip_x0=True, generator=generator, x0_guidance=x0_guidance
            )

        all_images.append(x)

    return torch.cat(all_images, dim=0)[:num_samples]


@torch.no_grad()
def sample_rfm_only(
    model,
    scheduler,
    rfm_directions: List[Tuple[torch.Tensor, float]],  # (direction, strength_mult)
    config: RFMConfig,
    num_samples: int = 256,
    batch_size: int = 8,
    seed: int = 42,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate samples with RFM-only guidance."""
    all_images = []
    generator = torch.Generator(device=device).manual_seed(seed)
    schedule = build_ddim_schedule_from_scheduler(scheduler, config.num_steps, device)

    # Create hooks with sigma window to stop RFM at low sigma (improves KID)
    hooks = []
    for direction, strength_mult in rfm_directions:
        hook = ActivationGuidanceHook(
            direction=direction.to(device),
            strength=config.rfm_strength * strength_mult,
            window=GuidanceWindow(
                start_sigma=config.rfm_sigma_end,  # Stop at this sigma
                end_sigma=1000.0,                   # No upper limit
            ),
            guidance_type="additive",
            adaptive_strength=True,
        )
        hook.register(model, "mid_block")
        hooks.append(hook)

    num_batches = (num_samples + batch_size - 1) // batch_size

    try:
        for _ in tqdm(range(num_batches), desc="RFM-only sampling"):
            current_batch = min(batch_size, num_samples - len(all_images))

            x = torch.randn(
                (current_batch, 3, 256, 256),
                device=device,
                generator=generator,
            )

            for step_idx in range(len(schedule.timesteps)):
                t = schedule.timesteps[step_idx]
                alpha_prod_t = schedule.alpha_prod_t[step_idx]
                alpha_prod_t_prev = schedule.alpha_prod_t_prev[step_idx]
                sigma = schedule.sigmas[step_idx].item()

                t_batch = torch.full((x.shape[0],), int(t), device=device, dtype=torch.long)

                for hook in hooks:
                    hook.set_step(step_idx, sigma)

                # CFG at eps level
                for hook in hooks:
                    hook.enabled = True
                eps_guided = predict_eps(model, x, t_batch)

                for hook in hooks:
                    hook.enabled = False
                eps_unguided = predict_eps(model, x, t_batch)

                for hook in hooks:
                    hook.enabled = True

                eps = eps_unguided + config.rfm_cfg_scale * (eps_guided - eps_unguided)

                x = ddim_step(
                    x, eps, alpha_prod_t, alpha_prod_t_prev, config.eta,
                    clip_x0=True, generator=generator
                )

            all_images.append(x)
    finally:
        for hook in hooks:
            hook.remove()

    return torch.cat(all_images, dim=0)[:num_samples]


@torch.no_grad()
def sample_hybrid(
    model,
    scheduler,
    wiener_denoisers: List[Tuple[GaussianDenoiser, float]],
    full_denoiser: GaussianDenoiser,
    rfm_directions: List[Tuple[torch.Tensor, float]],
    config: HybridConfig,
    num_samples: int = 256,
    batch_size: int = 8,
    seed: int = 42,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate samples with hybrid Wiener + RFM guidance."""
    all_images = []
    generator = torch.Generator(device=device).manual_seed(seed)
    schedule = build_ddim_schedule_from_scheduler(scheduler, config.num_steps, device)

    # Create RFM hooks with sigma window
    hooks = []
    for direction, strength_mult in rfm_directions:
        hook = ActivationGuidanceHook(
            direction=direction.to(device),
            strength=config.rfm_strength * strength_mult,
            window=GuidanceWindow(
                start_sigma=config.rfm_sigma_end,  # Stop at this sigma (improves KID)
                end_sigma=config.rfm_sigma_start,
            ),
            guidance_type="additive",
            adaptive_strength=True,
        )
        hook.register(model, "mid_block")
        hooks.append(hook)

    num_batches = (num_samples + batch_size - 1) // batch_size

    try:
        for _ in tqdm(range(num_batches), desc="Hybrid sampling"):
            current_batch = min(batch_size, num_samples - len(all_images))

            x = torch.randn(
                (current_batch, 3, 256, 256),
                device=device,
                generator=generator,
            )

            for step_idx in range(len(schedule.timesteps)):
                t = schedule.timesteps[step_idx]
                alpha_prod_t = schedule.alpha_prod_t[step_idx]
                alpha_prod_t_prev = schedule.alpha_prod_t_prev[step_idx]
                sigma = schedule.sigmas[step_idx].item()
                ab_t = float(alpha_prod_t.item())

                t_batch = torch.full((x.shape[0],), int(t), device=device, dtype=torch.long)

                for hook in hooks:
                    hook.set_step(step_idx, sigma)

                use_rfm_cfg = sigma < config.rfm_sigma_start

                if use_rfm_cfg:
                    # Low sigma: RFM with CFG
                    for hook in hooks:
                        hook.enabled = True
                    eps_guided = predict_eps(model, x, t_batch)

                    for hook in hooks:
                        hook.enabled = False
                    eps_unguided = predict_eps(model, x, t_batch)

                    for hook in hooks:
                        hook.enabled = True

                    eps = eps_unguided + config.rfm_cfg_scale * (eps_guided - eps_unguided)

                    x = ddim_step(
                        x, eps, alpha_prod_t, alpha_prod_t_prev, config.eta,
                        clip_x0=True, generator=generator
                    )
                else:
                    # High sigma: Wiener guidance
                    eps = predict_eps(model, x, t_batch)

                    x0_guidance = None
                    if sigma >= config.wiener_sigma_end:
                        x_edm = x / np.sqrt(ab_t)
                        d_full = full_denoiser.denoise(x_edm.float(), sigma)

                        x0_guidance = torch.zeros_like(x)
                        for denoiser, strength in wiener_denoisers:
                            d_class = denoiser.denoise(x_edm.float(), sigma)
                            x0_guidance += strength * (d_class - d_full)
                        x0_guidance = x0_guidance.to(x.dtype)

                    x = ddim_step(
                        x, eps, alpha_prod_t, alpha_prod_t_prev, config.eta,
                        clip_x0=True, generator=generator, x0_guidance=x0_guidance
                    )

            all_images.append(x)
    finally:
        for hook in hooks:
            hook.remove()

    return torch.cat(all_images, dim=0)[:num_samples]


# =============================================================================
# Main Comparison
# =============================================================================

def run_comparison(
    model,
    scheduler,
    wiener_cache: Dict[str, GaussianDenoiser],
    full_denoiser: GaussianDenoiser,
    evaluators: dict,
    combo_name: str,
    attributes: List[str],
    num_samples: int = 256,
    seed: int = 42,
    device: str = "cuda",
    rfm_sigma_end: float = 0.5,
    wiener_strength: float = 2.0,
    eta: float = 0.0,
    batch_size: int = 8,
    hybrid_rfm_strength: float = 0.4,
    hybrid_rfm_sigma_start: float = 3.5,
    methods_to_run: str = "all",
    skip_quality_metrics: bool = False,
) -> Dict[str, dict]:
    """Run comparison for a single combination across all methods."""
    print(f"\n{'='*70}")
    print(f"COMPARING: {combo_name}")
    print(f"Attributes: {attributes}")
    print(f"{'='*70}")

    results = {}

    # Configs
    wiener_config = WienerConfig(wiener_strength=wiener_strength, eta=eta)
    rfm_config = RFMConfig(rfm_sigma_end=rfm_sigma_end, eta=eta)
    hybrid_config = HybridConfig(
        wiener_strength=wiener_strength,
        rfm_strength=hybrid_rfm_strength,
        rfm_sigma_start=hybrid_rfm_sigma_start,
        rfm_sigma_end=rfm_sigma_end,
        eta=eta
    )

    # Build Wiener denoisers
    # For nonblond: use blond filter with negative strength
    wiener_denoisers = []
    for attr in attributes:
        # For nonblond, use the blond multiplier (since nonblond uses blond filter negatively)
        mult_key = "blond" if attr == "nonblond" else attr
        strength_mult = WIENER_STRENGTH_MULT.get(mult_key, 1.0)
        if attr == "nonblond" and "blond" in wiener_cache:
            # Use negative blond for nonblond steering
            wiener_denoisers.append((wiener_cache["blond"], -wiener_config.wiener_strength * strength_mult))
            print(f"  Wiener: blond x -1 x {strength_mult} (for nonblond)")
        elif attr in wiener_cache:
            wiener_denoisers.append((wiener_cache[attr], wiener_config.wiener_strength * strength_mult))
            print(f"  Wiener: {attr} x {strength_mult}")

    # Build RFM directions
    rfm_directions = []
    for attr in attributes:
        concept_name, sign = RFM_CONCEPTS[attr]
        concept_file = f"{concept_name}_{BLOCK_NAME}_t{TIMESTEP}.pt"
        concept_path = CONCEPTS_DIR / concept_file

        if concept_path.exists():
            direction, auc = load_concept_direction(concept_path, topk=rfm_config.topk, device=device)
            direction = direction * sign
            strength_mult = RFM_STRENGTH_MULT.get(attr, 1.0)
            rfm_directions.append((direction, strength_mult))
            print(f"  Loaded RFM: {concept_name} x {sign:+d} (AUC={auc:.4f})")
        else:
            print(f"  Missing RFM: {concept_path}")

    # Parse methods to run
    run_wiener = methods_to_run in ["all", "wiener"]
    run_rfm = methods_to_run in ["all", "rfm"]
    run_hybrid = methods_to_run in ["all", "hybrid"]

    # === 1. Wiener-only ===
    if wiener_denoisers and run_wiener:
        print(f"\n--- Wiener-only ---")
        samples = sample_wiener_only(
            model, scheduler, wiener_denoisers, full_denoiser,
            wiener_config, num_samples=num_samples, batch_size=batch_size, seed=seed, device=device
        )
        images = tensor_to_pil(samples)

        acc_results = evaluate_combo(evaluators, images, combo_name)
        kid, inception = compute_quality_metrics(images, device, skip=skip_quality_metrics)

        results["wiener"] = {
            **acc_results,
            "log_kid": float(np.log(kid)),
            "inception_score": inception,
        }
        print(f"  Accuracy: {acc_results}")
        print(f"  log-KID: {np.log(kid):.4f}, IS: {inception:.2f}")

        # Save grid
        grid = make_image_grid(samples[:64], nrow=8)
        grid.save(RESULTS_DIR / f"grid_{combo_name}_wiener.png")
        torch.save(samples.cpu(), RESULTS_DIR / f"samples_{combo_name}_wiener.pt")
    else:
        print(f"\n--- Wiener-only: SKIPPED (no filters for {attributes}) ---")
        results["wiener"] = None

    # === 2. RFM-only ===
    if rfm_directions and run_rfm:
        print(f"\n--- RFM-only ---")
        samples = sample_rfm_only(
            model, scheduler, rfm_directions,
            rfm_config, num_samples=num_samples, batch_size=batch_size, seed=seed, device=device
        )
        images = tensor_to_pil(samples)

        acc_results = evaluate_combo(evaluators, images, combo_name)
        kid, inception = compute_quality_metrics(images, device, skip=skip_quality_metrics)

        results["rfm"] = {
            **acc_results,
            "log_kid": float(np.log(kid)),
            "inception_score": inception,
        }
        print(f"  Accuracy: {acc_results}")
        print(f"  log-KID: {np.log(kid):.4f}, IS: {inception:.2f}")

        grid = make_image_grid(samples[:64], nrow=8)
        grid.save(RESULTS_DIR / f"grid_{combo_name}_rfm.png")
        torch.save(samples.cpu(), RESULTS_DIR / f"samples_{combo_name}_rfm.pt")
    else:
        print(f"\n--- RFM-only: SKIPPED (no concepts for {attributes}) ---")
        results["rfm"] = None

    # === 3. Hybrid ===
    if wiener_denoisers and rfm_directions and run_hybrid:
        print(f"\n--- Hybrid ---")
        samples = sample_hybrid(
            model, scheduler, wiener_denoisers, full_denoiser, rfm_directions,
            hybrid_config, num_samples=num_samples, batch_size=batch_size, seed=seed, device=device
        )
        images = tensor_to_pil(samples)

        acc_results = evaluate_combo(evaluators, images, combo_name)
        kid, inception = compute_quality_metrics(images, device, skip=skip_quality_metrics)

        results["hybrid"] = {
            **acc_results,
            "log_kid": float(np.log(kid)),
            "inception_score": inception,
        }
        print(f"  Accuracy: {acc_results}")
        print(f"  log-KID: {np.log(kid):.4f}, IS: {inception:.2f}")

        grid = make_image_grid(samples[:64], nrow=8)
        grid.save(RESULTS_DIR / f"grid_{combo_name}_hybrid.png")
        torch.save(samples.cpu(), RESULTS_DIR / f"samples_{combo_name}_hybrid.pt")
    else:
        print(f"\n--- Hybrid: SKIPPED ---")
        results["hybrid"] = None

    return results


def main():
    parser = argparse.ArgumentParser(description="ICML Comparison: Wiener vs RFM vs Hybrid")
    parser.add_argument("--quick", action="store_true", help="Quick test (16 samples, male_blond only)")
    parser.add_argument("--baseline-only", action="store_true", help="Generate baseline only")
    parser.add_argument("--num-samples", type=int, default=256, help="Number of samples per method")
    parser.add_argument("--combo", type=str, default=None, help="Run single combo only")
    # Tuning args for sweep
    parser.add_argument("--rfm-sigma-end", type=float, default=0.5, help="Stop RFM when sigma < this (default: 0.5)")
    parser.add_argument("--rfm-male-mult", type=float, default=1.0, help="RFM strength multiplier for male (default: 1.0)")
    parser.add_argument("--rfm-old-mult", type=float, default=1.0, help="RFM strength multiplier for old (default: 1.0)")
    parser.add_argument("--rfm-blond-mult", type=float, default=1.0, help="RFM strength multiplier for blond (default: 1.0)")
    parser.add_argument("--rfm-female-mult", type=float, default=0.5, help="RFM strength multiplier for female (default: 0.5)")
    parser.add_argument("--wiener-strength", type=float, default=2.0, help="Wiener guidance strength (default: 2.0)")
    parser.add_argument("--wiener-male-mult", type=float, default=1.0, help="Wiener strength multiplier for male (default: 1.0)")
    parser.add_argument("--wiener-old-mult", type=float, default=1.0, help="Wiener strength multiplier for old (default: 1.0)")
    parser.add_argument("--wiener-blond-mult", type=float, default=1.0, help="Wiener strength multiplier for blond (default: 1.0)")
    parser.add_argument("--wiener-female-mult", type=float, default=1.0, help="Wiener strength multiplier for female (default: 1.0)")
    parser.add_argument("--wiener-young-mult", type=float, default=1.0, help="Wiener strength multiplier for young (default: 1.0)")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta (0.0=deterministic, 1.0=stochastic)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for sampling (default: 8, increase for faster generation)")
    # Hybrid-specific args
    parser.add_argument("--hybrid-rfm-strength", type=float, default=0.4, help="RFM strength for hybrid method (default: 0.4)")
    parser.add_argument("--hybrid-rfm-sigma-start", type=float, default=3.5, help="Transition point from Wiener to RFM (default: 3.5)")
    # Method selection
    parser.add_argument("--methods", type=str, default="all", help="Methods to run: 'wiener', 'rfm', 'hybrid', or 'all' (default: all)")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--skip-quality-metrics", action="store_true", help="Skip KID/Inception Score and compute classifier accuracies only")
    args = parser.parse_args()

    # Override RFM strength multipliers from CLI
    RFM_STRENGTH_MULT["male"] = args.rfm_male_mult
    RFM_STRENGTH_MULT["old"] = args.rfm_old_mult
    RFM_STRENGTH_MULT["blond"] = args.rfm_blond_mult
    RFM_STRENGTH_MULT["female"] = args.rfm_female_mult

    # Override Wiener strength multipliers from CLI
    WIENER_STRENGTH_MULT["male"] = args.wiener_male_mult
    WIENER_STRENGTH_MULT["old"] = args.wiener_old_mult
    WIENER_STRENGTH_MULT["blond"] = args.wiener_blond_mult
    WIENER_STRENGTH_MULT["female"] = args.wiener_female_mult
    WIENER_STRENGTH_MULT["young"] = args.wiener_young_mult

    # Override output directory if specified
    global RESULTS_DIR
    if args.output_dir:
        RESULTS_DIR = Path(args.output_dir)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ICML Comparison: Wiener vs RFM vs Hybrid")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Set sample count
    num_samples = 16 if args.quick else args.num_samples
    print(f"Samples per method: {num_samples}")

    # Print sweep params
    print(f"\nSweep parameters:")
    print(f"  wiener_strength: {args.wiener_strength}")
    print(f"  wiener_male_mult: {args.wiener_male_mult}")
    print(f"  wiener_old_mult: {args.wiener_old_mult}")
    print(f"  wiener_blond_mult: {args.wiener_blond_mult}")
    print(f"  wiener_female_mult: {args.wiener_female_mult}")
    print(f"  wiener_young_mult: {args.wiener_young_mult}")
    print(f"  rfm_sigma_end: {args.rfm_sigma_end}")
    print(f"  rfm_male_mult: {args.rfm_male_mult}")
    print(f"  rfm_old_mult: {args.rfm_old_mult}")
    print(f"  rfm_blond_mult: {args.rfm_blond_mult}")
    print(f"  rfm_female_mult: {args.rfm_female_mult}")
    print(f"  hybrid_rfm_strength: {args.hybrid_rfm_strength}")
    print(f"  hybrid_rfm_sigma_start: {args.hybrid_rfm_sigma_start}")
    print(f"  eta: {args.eta}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  methods: {args.methods}")
    print(f"  output_dir: {RESULTS_DIR}")
    print(f"  skip_quality_metrics: {args.skip_quality_metrics}")

    # Load model
    model, scheduler = load_celeba_model(device)

    # Load Wiener filters
    print("\nLoading Wiener filters...")
    full_filter = load_gaussian_filter(str(WIENER_FILTER_DIR / "celeba_full.pt"), device=device)
    full_denoiser = GaussianDenoiser(full_filter, denoising_mode="wiener").to(device)

    wiener_cache = {}
    for attr, filter_name in WIENER_FILTERS.items():
        filter_path = WIENER_FILTER_DIR / filter_name
        if filter_path.exists():
            filt = load_gaussian_filter(str(filter_path), device=device)
            wiener_cache[attr] = GaussianDenoiser(filt, denoising_mode="wiener").to(device)
            print(f"  Loaded: {attr}")

    # Load evaluators
    evaluators = load_evaluators(device)

    all_results = {}

    # === Baseline ===
    print("\n" + "=" * 70)
    print("BASELINE (no guidance)")
    print("=" * 70)
    baseline_samples = sample_baseline(model, scheduler, num_samples=num_samples, batch_size=args.batch_size, device=device)
    baseline_images = tensor_to_pil(baseline_samples)

    baseline_kid, baseline_is = compute_quality_metrics(
        baseline_images, device, skip=args.skip_quality_metrics
    )

    all_results["baseline"] = {
        "log_kid": float(np.log(baseline_kid)),
        "inception_score": baseline_is,
    }
    print(f"  log-KID: {np.log(baseline_kid):.4f}, IS: {baseline_is:.2f}")

    grid = make_image_grid(baseline_samples[:64], nrow=8)
    grid.save(RESULTS_DIR / "grid_baseline.png")

    if args.baseline_only:
        output_path = RESULTS_DIR / "comparison_results.json"
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nBaseline results saved to: {output_path}")
        return

    # === Run comparisons ===
    seed = 42

    if args.quick:
        combos_to_run = {"male_blond": COMBOS["male_blond"]}
    elif args.combo:
        combos_to_run = {args.combo: COMBOS[args.combo]}
    else:
        combos_to_run = COMBOS

    for combo_name, attributes in combos_to_run.items():
        combo_results = run_comparison(
            model, scheduler, wiener_cache, full_denoiser, evaluators,
            combo_name, attributes, num_samples=num_samples, seed=seed, device=device,
            rfm_sigma_end=args.rfm_sigma_end,
            wiener_strength=args.wiener_strength,
            eta=args.eta,
            batch_size=args.batch_size,
            hybrid_rfm_strength=args.hybrid_rfm_strength,
            hybrid_rfm_sigma_start=args.hybrid_rfm_sigma_start,
            methods_to_run=args.methods,
            skip_quality_metrics=args.skip_quality_metrics,
        )
        all_results[combo_name] = combo_results

    # Save results
    output_path = RESULTS_DIR / "comparison_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # === Generate summary table ===
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    # Header
    print(f"\n{'Combination':<16} {'Method':<10} {'Acc1':>8} {'Acc2':>8} {'Combined':>10} {'log-KID':>10} {'IS':>8}")
    print("-" * 80)

    # Baseline
    baseline = all_results.get("baseline", {})
    print(f"{'baseline':<16} {'-':<10} {'-':>8} {'-':>8} {'-':>10} {baseline.get('log_kid', 0):.4f}     {baseline.get('inception_score', 0):.2f}")
    print("-" * 80)

    for combo_name in combos_to_run.keys():
        combo_results = all_results.get(combo_name, {})

        for method in ["wiener", "rfm", "hybrid"]:
            method_results = combo_results.get(method)
            if method_results is None:
                continue

            # Get attribute accuracies
            acc_keys = [k for k in method_results.keys() if k not in ["combined", "log_kid", "inception_score"]]
            acc1 = method_results.get(acc_keys[0], 0) if acc_keys else 0
            acc2 = method_results.get(acc_keys[1], 0) if len(acc_keys) > 1 else 0
            combined = method_results.get("combined", acc1)
            log_kid = method_results.get("log_kid", 0)
            inception = method_results.get("inception_score", 0)

            print(f"{combo_name:<16} {method:<10} {100*acc1:>7.1f}% {100*acc2:>7.1f}% {100*combined:>9.1f}% {log_kid:>10.4f} {inception:>8.2f}")

        print("-" * 80)

    # Generate markdown table
    md_lines = [
        "| Combination | Method | Attr1 | Attr2 | Combined | log-KID | IS |",
        "|-------------|--------|-------|-------|----------|---------|-----|",
    ]

    # Baseline
    baseline = all_results.get("baseline", {})
    md_lines.append(f"| baseline | - | - | - | - | {baseline.get('log_kid', 0):.4f} | {baseline.get('inception_score', 0):.2f} |")

    for combo_name in combos_to_run.keys():
        combo_results = all_results.get(combo_name, {})

        for method in ["wiener", "rfm", "hybrid"]:
            method_results = combo_results.get(method)
            if method_results is None:
                continue

            acc_keys = [k for k in method_results.keys() if k not in ["combined", "log_kid", "inception_score"]]
            acc1 = method_results.get(acc_keys[0], 0) if acc_keys else 0
            acc2 = method_results.get(acc_keys[1], 0) if len(acc_keys) > 1 else 0
            combined = method_results.get("combined", acc1)
            log_kid = method_results.get("log_kid", 0)
            inception = method_results.get("inception_score", 0)

            md_lines.append(f"| {combo_name} | {method} | {100*acc1:.1f}% | {100*acc2:.1f}% | {100*combined:.1f}% | {log_kid:.4f} | {inception:.2f} |")

    table_path = RESULTS_DIR / "table.md"
    with open(table_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"\nTable saved to: {table_path}")


if __name__ == "__main__":
    main()
