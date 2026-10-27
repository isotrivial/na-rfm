#!/usr/bin/env python3
"""CelebA-HQ parameter sweep for the accuracy and quality trade-off.

Sweeps key parameters to find settings with both good accuracy AND good KID.

Key insight from quick tests:
- Baseline: log-KID = -2.83 (best quality)
- Guided methods: log-KID = -1.5 to -2.4 (quality degrades with stronger guidance)

Goal: Find parameters where we maintain log-KID < -2.0 while maximizing accuracy.

Parameters to sweep:
- Wiener: strength (0.5-3.0), sigma_end (2.0-5.0)
- RFM: strength (0.1-0.5), cfg_scale (1.0-3.0)
- Hybrid: wiener_strength, rfm_strength, rfm_cfg_scale

Usage:
    cd <repo-root>
    PYTHONPATH=".:third_party/edm:baselines/tfg:$PYTHONPATH" \
      python repro/celeba_parameter_sweep.py

    # Sweep specific method
    PYTHONPATH=".:third_party/edm:baselines/tfg:$PYTHONPATH" \
      python repro/celeba_parameter_sweep.py --method hybrid

    # Sweep specific combo
    PYTHONPATH=".:third_party/edm:baselines/tfg:$PYTHONPATH" \
      python repro/celeba_parameter_sweep.py --combo male_blond
"""

import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Optional, List, Tuple, Dict
import os
from itertools import product

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
    predict_eps,
    ddim_step,
)

from na_rfm.celeba_config import (
    CONCEPTS_DIR,
    BLOCK_NAME,
    TIMESTEP,
)


# Paths
RESULTS_DIR = PROJECT_ROOT / "results" / "celeba" / "parameter_sweep"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WIENER_FILTER_DIR = PROJECT_ROOT / "repro" / "assets" / "celeba_filters"
CELEBA_DATA_DIR = PROJECT_ROOT / "data" / "celeba_hq_256_30k"

CELEBA_DDPM_MODEL_ID = "google/ddpm-ema-celebahq-256"

# Evaluation classifiers
EVAL_GENDER_MODEL = "rizvandwiki/gender-classification"
EVAL_AGE_MODEL = "ibombonato/swin-age-classifier"
EVAL_HAIR_MODEL = "londe33/hair_v02"

# Wiener filters
WIENER_FILTERS = {
    "male": "celeba_male.pt",
    "female": "celeba_female.pt",
    "young": "celeba_young.pt",
    "old": "celeba_old.pt",
    "blond": "celeba_blond_hair.pt",
}

# RFM concepts
RFM_CONCEPTS = {
    "male": ("male_vs_female", +1),
    "female": ("male_vs_female", -1),
    "young": ("young_vs_other", +1),
    "old": ("old_vs_other", +1),
    "blond": ("blond_vs_other", +1),
    "nonblond": ("blond_vs_other", -1),
}

RFM_STRENGTH_MULT = {
    "male": 1.0,
    "female": 0.5,
    "young": 0.8,
    "old": 1.0,
    "blond": 1.0,
    "nonblond": 1.0,
}

COMBOS = {
    "male_blond": ["male", "blond"],
    "male_nonblond": ["male", "nonblond"],
    "female_blond": ["female", "blond"],
    "female_nonblond": ["female", "nonblond"],
    "young_male": ["young", "male"],
    "young_female": ["young", "female"],
    "old_male": ["old", "male"],
    "old_female": ["old", "female"],
}

# =============================================================================
# Sweep Configurations
# =============================================================================

WIENER_SWEEP = {
    "strength": [0.5, 1.0, 1.5, 2.0],
    "sigma_end": [2.0, 3.5, 5.0],
}

RFM_SWEEP = {
    "strength": [0.1, 0.2, 0.3],
    "cfg_scale": [1.5, 2.0, 2.5],
}

HYBRID_SWEEP = {
    "wiener_strength": [1.0, 1.5, 2.0],
    "rfm_strength": [0.2, 0.3, 0.4],
    "rfm_cfg_scale": [1.5, 2.0],
}


# =============================================================================
# Model & Data Loading (reused from 06)
# =============================================================================

def load_celeba_model(device: str = "cuda"):
    from diffusers import DDIMPipeline
    print(f"Loading CelebA DDPM model: {CELEBA_DDPM_MODEL_ID}")
    pipe = DDIMPipeline.from_pretrained(CELEBA_DDPM_MODEL_ID)
    pipe.unet.to(device).eval()
    return pipe.unet, pipe.scheduler


def load_concept_direction(path: Path, topk: int = 5, device: str = "cuda"):
    data = torch.load(path, weights_only=False)
    eigenvectors = data["eigenvectors"].to(device)
    eigenvalues = data["eigenvalues"].to(device)
    sign = data["sign"]

    k = min(topk, len(eigenvalues))
    weights = eigenvalues[:k] / (eigenvalues[:k].sum() + 1e-12)
    direction = (eigenvectors[:, :k] * weights.unsqueeze(0)).sum(dim=1)
    direction = direction / (torch.norm(direction) + 1e-8)
    direction = direction * sign

    return direction, data.get("auc", None)


def load_celeba_reference_images(num_samples: int = 1000) -> List[Image.Image]:
    images = []
    img_files = sorted(os.listdir(CELEBA_DATA_DIR))[:num_samples]
    for img_file in img_files:
        img_path = CELEBA_DATA_DIR / img_file
        images.append(Image.open(img_path).convert("RGB"))
    return images


# =============================================================================
# Metrics (reused from 06)
# =============================================================================

@torch.no_grad()
def compute_kid(generated_images: List[Image.Image], ref_images: List[Image.Image], device: str = "cuda") -> float:
    kid = KernelInceptionDistance(subset_size=min(len(generated_images), 50), normalize=True).to(device)

    def image_proc(images):
        tf = transforms.Compose([transforms.Resize(299), transforms.ToTensor()])
        return torch.stack([tf(img) for img in images], dim=0).to(device)

    batch_size = 32
    for i in range(0, len(ref_images), batch_size):
        ref_batch = ref_images[i:i+batch_size]
        kid.update(image_proc(ref_batch), real=True)

    for i in range(0, len(generated_images), batch_size):
        batch = generated_images[i:i+batch_size]
        kid.update(image_proc(batch), real=False)

    score = kid.compute()
    return score[0].item()


def load_evaluators(device: str = "cuda") -> dict:
    evaluators = {}
    print(f"Loading gender evaluator: {EVAL_GENDER_MODEL}")
    evaluators["gender"] = pipeline("image-classification", model=EVAL_GENDER_MODEL, device=0 if device == "cuda" else -1)
    print(f"Loading age evaluator: {EVAL_AGE_MODEL}")
    evaluators["age"] = pipeline("image-classification", model=EVAL_AGE_MODEL, device=0 if device == "cuda" else -1)
    print(f"Loading hair evaluator: {EVAL_HAIR_MODEL}")
    evaluators["hair"] = pipeline("image-classification", model=EVAL_HAIR_MODEL, device=0 if device == "cuda" else -1)
    return evaluators


def evaluate_gender(evaluator, images: list, target: str) -> float:
    results = evaluator(images, top_k=2)
    correct = 0
    for result in results:
        top_label = result[0]["label"].lower()
        if target == "female":
            if "female" in top_label or "woman" in top_label:
                correct += 1
        else:
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
        is_blond = "blond" in top_label or "blonde" in top_label or "yellow" in top_label
        if target == "blond" and is_blond:
            correct += 1
        elif target == "nonblond" and "black" in top_label:
            correct += 1
    return correct / len(images)


def evaluate_combo(evaluators: dict, images: list, combo_name: str) -> Dict[str, float]:
    results = {}
    if "male" in combo_name and "female" not in combo_name:
        results["gender"] = evaluate_gender(evaluators["gender"], images, "male")
    elif "female" in combo_name:
        results["gender"] = evaluate_gender(evaluators["gender"], images, "female")
    if "young" in combo_name:
        results["age"] = evaluate_age(evaluators["age"], images, "young")
    elif "old" in combo_name:
        results["age"] = evaluate_age(evaluators["age"], images, "old")
    if "blond" in combo_name and "nonblond" not in combo_name:
        results["hair"] = evaluate_hair(evaluators["hair"], images, "blond")
    elif "nonblond" in combo_name:
        results["hair"] = evaluate_hair(evaluators["hair"], images, "nonblond")
    if len(results) >= 2:
        results["combined"] = float(np.mean(list(results.values())))
    return results


def tensor_to_pil(samples: torch.Tensor) -> list:
    samples = ((samples + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    samples = samples.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(img) for img in samples]


# =============================================================================
# Sampling Functions (simplified for sweep)
# =============================================================================

@torch.no_grad()
def sample_wiener(model, scheduler, wiener_denoisers, full_denoiser,
                  strength, sigma_end, num_samples, num_steps=50, seed=42, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(seed)
    schedule = build_ddim_schedule_from_scheduler(scheduler, num_steps, device)

    all_images = []
    batch_size = 8
    num_batches = (num_samples + batch_size - 1) // batch_size

    for _ in range(num_batches):
        current_batch = min(batch_size, num_samples - len(all_images))
        x = torch.randn((current_batch, 3, 256, 256), device=device, generator=generator)

        for step_idx in range(len(schedule.timesteps)):
            t = schedule.timesteps[step_idx]
            alpha_prod_t = schedule.alpha_prod_t[step_idx]
            alpha_prod_t_prev = schedule.alpha_prod_t_prev[step_idx]
            sigma = schedule.sigmas[step_idx].item()
            ab_t = float(alpha_prod_t.item())

            t_batch = torch.full((x.shape[0],), int(t), device=device, dtype=torch.long)
            eps = predict_eps(model, x, t_batch)

            x0_guidance = None
            if sigma >= sigma_end:
                x_edm = x / np.sqrt(ab_t)
                d_full = full_denoiser.denoise(x_edm.float(), sigma)
                x0_guidance = torch.zeros_like(x)
                for denoiser, base_strength in wiener_denoisers:
                    d_class = denoiser.denoise(x_edm.float(), sigma)
                    # Scale by sweep strength relative to base
                    x0_guidance += strength * np.sign(base_strength) * (d_class - d_full)
                x0_guidance = x0_guidance.to(x.dtype)

            x = ddim_step(x, eps, alpha_prod_t, alpha_prod_t_prev, 0.0, True, generator, x0_guidance)

        all_images.append(x)

    return torch.cat(all_images, dim=0)[:num_samples]


@torch.no_grad()
def sample_rfm(model, scheduler, rfm_directions, strength, cfg_scale,
               num_samples, num_steps=100, seed=42, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(seed)
    schedule = build_ddim_schedule_from_scheduler(scheduler, num_steps, device)

    hooks = []
    for direction, strength_mult in rfm_directions:
        hook = ActivationGuidanceHook(
            direction=direction.to(device),
            strength=strength * strength_mult,
            guidance_type="additive",
            adaptive_strength=True,
        )
        hook.register(model, "mid_block")
        hooks.append(hook)

    all_images = []
    batch_size = 8
    num_batches = (num_samples + batch_size - 1) // batch_size

    try:
        for _ in range(num_batches):
            current_batch = min(batch_size, num_samples - len(all_images))
            x = torch.randn((current_batch, 3, 256, 256), device=device, generator=generator)

            for step_idx in range(len(schedule.timesteps)):
                t = schedule.timesteps[step_idx]
                alpha_prod_t = schedule.alpha_prod_t[step_idx]
                alpha_prod_t_prev = schedule.alpha_prod_t_prev[step_idx]
                sigma = schedule.sigmas[step_idx].item()

                t_batch = torch.full((x.shape[0],), int(t), device=device, dtype=torch.long)

                for hook in hooks:
                    hook.set_step(step_idx, sigma)
                    hook.enabled = True
                eps_guided = predict_eps(model, x, t_batch)

                for hook in hooks:
                    hook.enabled = False
                eps_unguided = predict_eps(model, x, t_batch)

                for hook in hooks:
                    hook.enabled = True

                eps = eps_unguided + cfg_scale * (eps_guided - eps_unguided)
                x = ddim_step(x, eps, alpha_prod_t, alpha_prod_t_prev, 0.0, True, generator)

            all_images.append(x)
    finally:
        for hook in hooks:
            hook.remove()

    return torch.cat(all_images, dim=0)[:num_samples]


@torch.no_grad()
def sample_hybrid(model, scheduler, wiener_denoisers, full_denoiser, rfm_directions,
                  wiener_strength, rfm_strength, rfm_cfg_scale, sigma_transition=3.5,
                  num_samples=64, num_steps=100, seed=42, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(seed)
    schedule = build_ddim_schedule_from_scheduler(scheduler, num_steps, device)

    hooks = []
    for direction, strength_mult in rfm_directions:
        hook = ActivationGuidanceHook(
            direction=direction.to(device),
            strength=rfm_strength * strength_mult,
            window=GuidanceWindow(start_sigma=0.0, end_sigma=sigma_transition),
            guidance_type="additive",
            adaptive_strength=True,
        )
        hook.register(model, "mid_block")
        hooks.append(hook)

    all_images = []
    batch_size = 8
    num_batches = (num_samples + batch_size - 1) // batch_size

    try:
        for _ in range(num_batches):
            current_batch = min(batch_size, num_samples - len(all_images))
            x = torch.randn((current_batch, 3, 256, 256), device=device, generator=generator)

            for step_idx in range(len(schedule.timesteps)):
                t = schedule.timesteps[step_idx]
                alpha_prod_t = schedule.alpha_prod_t[step_idx]
                alpha_prod_t_prev = schedule.alpha_prod_t_prev[step_idx]
                sigma = schedule.sigmas[step_idx].item()
                ab_t = float(alpha_prod_t.item())

                t_batch = torch.full((x.shape[0],), int(t), device=device, dtype=torch.long)

                for hook in hooks:
                    hook.set_step(step_idx, sigma)

                if sigma < sigma_transition:
                    # RFM phase
                    for hook in hooks:
                        hook.enabled = True
                    eps_guided = predict_eps(model, x, t_batch)

                    for hook in hooks:
                        hook.enabled = False
                    eps_unguided = predict_eps(model, x, t_batch)

                    for hook in hooks:
                        hook.enabled = True

                    eps = eps_unguided + rfm_cfg_scale * (eps_guided - eps_unguided)
                    x = ddim_step(x, eps, alpha_prod_t, alpha_prod_t_prev, 0.0, True, generator)
                else:
                    # Wiener phase
                    eps = predict_eps(model, x, t_batch)

                    x0_guidance = None
                    x_edm = x / np.sqrt(ab_t)
                    d_full = full_denoiser.denoise(x_edm.float(), sigma)
                    x0_guidance = torch.zeros_like(x)
                    for denoiser, base_strength in wiener_denoisers:
                        d_class = denoiser.denoise(x_edm.float(), sigma)
                        x0_guidance += wiener_strength * np.sign(base_strength) * (d_class - d_full)
                    x0_guidance = x0_guidance.to(x.dtype)

                    x = ddim_step(x, eps, alpha_prod_t, alpha_prod_t_prev, 0.0, True, generator, x0_guidance)

            all_images.append(x)
    finally:
        for hook in hooks:
            hook.remove()

    return torch.cat(all_images, dim=0)[:num_samples]


# =============================================================================
# Sweep Functions
# =============================================================================

def sweep_wiener(model, scheduler, wiener_denoisers, full_denoiser, evaluators,
                 ref_images, combo_name, num_samples=64, device="cuda"):
    """Sweep Wiener parameters."""
    results = []

    configs = list(product(WIENER_SWEEP["strength"], WIENER_SWEEP["sigma_end"]))
    print(f"\nSweeping Wiener: {len(configs)} configurations")

    for strength, sigma_end in tqdm(configs, desc="Wiener sweep"):
        samples = sample_wiener(
            model, scheduler, wiener_denoisers, full_denoiser,
            strength=strength, sigma_end=sigma_end, num_samples=num_samples, device=device
        )
        images = tensor_to_pil(samples)

        acc = evaluate_combo(evaluators, images, combo_name)
        kid = compute_kid(images, ref_images, device)
        log_kid = float(np.log(kid))

        result = {
            "method": "wiener",
            "strength": strength,
            "sigma_end": sigma_end,
            **acc,
            "log_kid": log_kid,
        }
        results.append(result)

        print(f"  str={strength}, sigma_end={sigma_end}: combined={acc.get('combined', acc.get('gender', 0))*100:.1f}%, log_kid={log_kid:.3f}")

    return results


def sweep_rfm(model, scheduler, rfm_directions, evaluators, ref_images,
              combo_name, num_samples=64, device="cuda"):
    """Sweep RFM parameters."""
    results = []

    configs = list(product(RFM_SWEEP["strength"], RFM_SWEEP["cfg_scale"]))
    print(f"\nSweeping RFM: {len(configs)} configurations")

    for strength, cfg_scale in tqdm(configs, desc="RFM sweep"):
        samples = sample_rfm(
            model, scheduler, rfm_directions,
            strength=strength, cfg_scale=cfg_scale, num_samples=num_samples, device=device
        )
        images = tensor_to_pil(samples)

        acc = evaluate_combo(evaluators, images, combo_name)
        kid = compute_kid(images, ref_images, device)
        log_kid = float(np.log(kid))

        result = {
            "method": "rfm",
            "strength": strength,
            "cfg_scale": cfg_scale,
            **acc,
            "log_kid": log_kid,
        }
        results.append(result)

        print(f"  str={strength}, cfg={cfg_scale}: combined={acc.get('combined', acc.get('gender', 0))*100:.1f}%, log_kid={log_kid:.3f}")

    return results


def sweep_hybrid(model, scheduler, wiener_denoisers, full_denoiser, rfm_directions,
                 evaluators, ref_images, combo_name, num_samples=64, device="cuda"):
    """Sweep Hybrid parameters."""
    results = []

    configs = list(product(
        HYBRID_SWEEP["wiener_strength"],
        HYBRID_SWEEP["rfm_strength"],
        HYBRID_SWEEP["rfm_cfg_scale"]
    ))
    print(f"\nSweeping Hybrid: {len(configs)} configurations")

    for wiener_str, rfm_str, rfm_cfg in tqdm(configs, desc="Hybrid sweep"):
        samples = sample_hybrid(
            model, scheduler, wiener_denoisers, full_denoiser, rfm_directions,
            wiener_strength=wiener_str, rfm_strength=rfm_str, rfm_cfg_scale=rfm_cfg,
            num_samples=num_samples, device=device
        )
        images = tensor_to_pil(samples)

        acc = evaluate_combo(evaluators, images, combo_name)
        kid = compute_kid(images, ref_images, device)
        log_kid = float(np.log(kid))

        result = {
            "method": "hybrid",
            "wiener_strength": wiener_str,
            "rfm_strength": rfm_str,
            "rfm_cfg_scale": rfm_cfg,
            **acc,
            "log_kid": log_kid,
        }
        results.append(result)

        print(f"  w_str={wiener_str}, r_str={rfm_str}, r_cfg={rfm_cfg}: combined={acc.get('combined', acc.get('gender', 0))*100:.1f}%, log_kid={log_kid:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for accuracy vs quality")
    parser.add_argument("--method", type=str, default=None, choices=["wiener", "rfm", "hybrid"],
                        help="Sweep only this method")
    parser.add_argument("--combo", type=str, default="male_blond",
                        help="Combination to test")
    parser.add_argument("--num-samples", type=int, default=64,
                        help="Samples per configuration")
    args = parser.parse_args()

    print("=" * 70)
    print("Parameter Sweep: Accuracy vs Quality Trade-off")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Combo: {args.combo}")
    print(f"Samples per config: {args.num_samples}")

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

    # Load evaluators
    evaluators = load_evaluators(device)

    # Load reference images for KID
    print("\nLoading reference images for KID...")
    ref_images = load_celeba_reference_images(1000)

    # Build denoisers/directions for combo
    attributes = COMBOS[args.combo]

    wiener_denoisers = []
    for attr in attributes:
        if attr == "nonblond" and "blond" in wiener_cache:
            wiener_denoisers.append((wiener_cache["blond"], -1.0))
        elif attr in wiener_cache:
            wiener_denoisers.append((wiener_cache[attr], 1.0))

    rfm_directions = []
    for attr in attributes:
        concept_name, sign = RFM_CONCEPTS[attr]
        concept_path = CONCEPTS_DIR / f"{concept_name}_{BLOCK_NAME}_t{TIMESTEP}.pt"
        if concept_path.exists():
            direction, _ = load_concept_direction(concept_path, topk=5, device=device)
            direction = direction * sign
            strength_mult = RFM_STRENGTH_MULT.get(attr, 1.0)
            rfm_directions.append((direction, strength_mult))

    all_results = []

    # Run sweeps
    methods = [args.method] if args.method else ["wiener", "rfm", "hybrid"]

    for method in methods:
        if method == "wiener" and wiener_denoisers:
            results = sweep_wiener(
                model, scheduler, wiener_denoisers, full_denoiser, evaluators,
                ref_images, args.combo, args.num_samples, device
            )
            all_results.extend(results)

        elif method == "rfm" and rfm_directions:
            results = sweep_rfm(
                model, scheduler, rfm_directions, evaluators,
                ref_images, args.combo, args.num_samples, device
            )
            all_results.extend(results)

        elif method == "hybrid" and wiener_denoisers and rfm_directions:
            results = sweep_hybrid(
                model, scheduler, wiener_denoisers, full_denoiser, rfm_directions,
                evaluators, ref_images, args.combo, args.num_samples, device
            )
            all_results.extend(results)

    # Save results
    output_path = RESULTS_DIR / f"sweep_{args.combo}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Find best configs (Pareto optimal: high accuracy, low KID)
    print("\n" + "=" * 70)
    print("BEST CONFIGURATIONS (accuracy > 50%, sorted by log-KID)")
    print("=" * 70)

    good_results = [r for r in all_results if r.get("combined", r.get("gender", 0)) > 0.5]
    good_results.sort(key=lambda x: x["log_kid"])

    print(f"\n{'Method':<10} {'Params':<30} {'Combined':>10} {'log-KID':>10}")
    print("-" * 65)

    for r in good_results[:10]:
        method = r["method"]
        if method == "wiener":
            params = f"str={r['strength']}, sigma={r['sigma_end']}"
        elif method == "rfm":
            params = f"str={r['strength']}, cfg={r['cfg_scale']}"
        else:
            params = f"w={r['wiener_strength']}, r={r['rfm_strength']}, cfg={r['rfm_cfg_scale']}"

        combined = r.get("combined", r.get("gender", 0))
        print(f"{method:<10} {params:<30} {combined*100:>9.1f}% {r['log_kid']:>10.3f}")

    # Also find best accuracy configs
    print("\n" + "=" * 70)
    print("BEST ACCURACY (sorted by combined accuracy)")
    print("=" * 70)

    all_results.sort(key=lambda x: -x.get("combined", x.get("gender", 0)))

    print(f"\n{'Method':<10} {'Params':<30} {'Combined':>10} {'log-KID':>10}")
    print("-" * 65)

    for r in all_results[:10]:
        method = r["method"]
        if method == "wiener":
            params = f"str={r['strength']}, sigma={r['sigma_end']}"
        elif method == "rfm":
            params = f"str={r['strength']}, cfg={r['cfg_scale']}"
        else:
            params = f"w={r['wiener_strength']}, r={r['rfm_strength']}, cfg={r['rfm_cfg_scale']}"

        combined = r.get("combined", r.get("gender", 0))
        print(f"{method:<10} {params:<30} {combined*100:>9.1f}% {r['log_kid']:>10.3f}")


if __name__ == "__main__":
    main()
