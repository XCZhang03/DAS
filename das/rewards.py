from PIL import Image
import os
from pathlib import Path
import io
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from transformers import pipeline
from diffusers.utils import load_image
from importlib import resources
# ASSETS_PATH = resources.files("assets")

def jpeg_compressibility(inference_dtype=None, device=None):
    import io
    import numpy as np
    def loss_fn(images):
        if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        loss = torch.tensor(sizes, dtype=inference_dtype, device=device)
        rewards = -1 * loss

        return loss, rewards

    return loss_fn

def clip_score(
    inference_dtype=None, 
    device=None, 
    return_loss=False, 
):
    from das.scorers.clip_scorer import CLIPScorer

    scorer = CLIPScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = - scores
            return loss, scores

        return loss_fn

def aesthetic_score(
    torch_dtype=None,
    aesthetic_target=None,
    grad_scale=0,
    device=None,
    return_loss=False,
):
    from das.scorers.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images)

            if aesthetic_target is None: # default maximization
                loss = -1 * scores
            else:
                # using L1 to keep on same scale
                loss = abs(scores - aesthetic_target)
            return loss * grad_scale, scores

        return loss_fn


def hps_score(
    inference_dtype=None, 
    device=None, 
    return_loss=False, 
):
    from das.scorers.hpsv2_scorer import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = 1.0 - scores
            return loss, scores

        return loss_fn


def ImageReward(
    inference_dtype=None, 
    device=None, 
    return_loss=False, 
):
    # from das.scorers.ImageReward_scorer import ImageRewardScorer

    # scorer = ImageRewardScorer(dtype=torch.float32, device=device)
    # scorer.requires_grad_(True)

    # pip install image-reward
    import ImageReward as RM
    model = RM.load("ImageReward-v1.0", device=device)
    def _transform():
        return Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            images = _transform()(images).to(device)
            prompt_input = model.blip.tokenizer(
                prompts,
                padding='max_length',
                truncation=True,
                max_length=35,
                return_tensors="pt"
            ).to(device)
            scores = model.score_gard(prompt_input.input_ids, prompt_input.attention_mask, images)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            images = _transform()(images).to(device)
            prompt_input = model.blip.tokenizer(
                prompts,
                padding='max_length',
                truncation=True,
                max_length=35,
                return_tensors="pt"
            ).to(device)
            scores = model.score_gard(prompt_input.input_ids, prompt_input.attention_mask, images)
            

            loss = - scores
            return loss, scores

        return loss_fn
    


def PickScore(
    inference_dtype=None, 
    device=None, 
    return_loss=False, 
):
    from das.scorers.PickScore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = - scores
            return loss, scores

        return loss_fn