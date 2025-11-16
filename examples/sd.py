from diffusers import DiffusionPipeline, DDIMScheduler
from tqdm import tqdm
from das.diffusers_patch.pipeline_using_SMC import pipeline_using_smc
import torch
import numpy as np
import das.rewards as rewards
from PIL import Image
import os
from tqdm import tqdm

################### Configuration ###################
kl_coeff = 0.07
n_steps = 100
num_particles = 4
batch_p = 1
tempering_gamma = 0.008

prompt = "cat and a dog"
repeated_prompts = [prompt] * batch_p

# reward_fn = rewards.aesthetic_score(device = 'cuda')
reward_fn = rewards.ImageReward(device = 'cuda:6')

hps_reward = rewards.hps_score(device = 'cuda:6')



################### Initialize ###################
log_dir_sd_smc = "logs/DAS_SD/pick/qualitative"
os.makedirs(log_dir_sd_smc, exist_ok=True)

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(n_steps)
pipe.to("cuda:6", dtype=torch.float16)

total_rewards = []
total_hps = []

import json
with open("/home/linhw/code/DAS/benchmark_ir.json", "r") as f:
    benchmark_ir = json.load(f)

for prompt_id, item in enumerate(tqdm(benchmark_ir)):
    prompt = item["prompt"]
    repeated_prompts = [prompt] * batch_p

    image_reward_fn = lambda images: reward_fn(
                        images, 
                        repeated_prompts
                    )
    hps_reward_fn = lambda images: hps_reward(
                    images, 
                    repeated_prompts
                )
    ################### Inference ###################
    with torch.autocast('cuda'):
        image = pipeline_using_smc(
            pipe,
            prompt=prompt,
            negative_prompt="",
            num_inference_steps=n_steps,
            output_type="pt",
            # SMC parameters
            num_particles=num_particles,
            batch_p=batch_p,
            tempering_gamma=tempering_gamma,
            reward_fn=image_reward_fn,
            kl_coeff=kl_coeff,
            verbose=False
        )[0]
    
    with torch.no_grad():
        image = image.to(torch.float32)
        reward = image_reward_fn(image).item()
        hps = hps_reward_fn(image).item()
    image = (image[0].cpu().numpy() * 255).transpose(1, 2, 0).round().astype(np.uint8)
    image = Image.fromarray(image)
    image.save(f"{log_dir_sd_smc}/{prompt[:20]} | reward: {reward}.png")
    print(f"Saved in {log_dir_sd_smc}/{prompt[:20]} | reward: {reward}.png | HPS: {hps}")
    total_rewards.append(reward)
    total_hps.append(hps)

print(f"Average reward: {np.mean(total_rewards)}")
print(f"Average hps: {np.mean(total_hps)}")

with open(f"{log_dir_sd_smc}/results.txt", "w") as f:
    f.write(f"Average reward: {np.mean(total_rewards)}\n")
    f.write(f"Average hps: {np.mean(total_hps)}\n")
    f.write(f"tempering gamma: {tempering_gamma}\n")
    f.write(f"kl coeff: {kl_coeff}\n")
    f.write(f"num particles: {num_particles}\n")
