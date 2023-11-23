#!/usr/bin/env python
import argparse
import datetime
import os
import random
import sys
import torch

from diffusers import StableDiffusionXLPipeline
from compel import Compel, ReturnedEmbeddingsType


def setup_pipe(args):
    device = args.device
    use_refiner = args.refiner

    common_params = {
        "use_safetensors": True,
    } if device == "cpu" else {
        "torch_dtype": torch.float16,
        "use_safetensors": True,
        "variant": "fp16",
    }

    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", **common_params)
    pipe = pipe.to(device)

    if use_refiner:
        refiner_params = {
            "unet": pipe.unet,
            "text_encoder_2": pipe.text_encoder_2,
            "vae": pipe.vae,
            **common_params,
        }
        refiner = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", **refiner_params)


        refiner = refiner.to(device)
        if device == 'cuda':
            pipe.enable_model_cpu_offload()
    else:
        refiner = None

    return pipe, refiner


def process_prompts(pipe, prompt, neg_prompt):
    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True]
    )        

    base_positive_prompt_embeds, base_positive_prompt_pooled = compel(prompt)
    base_negative_prompt_embeds, base_negative_prompt_pooled = compel(neg_prompt)

    base_positive_prompt_embeds, base_negative_prompt_embeds = compel.pad_conditioning_tensors_to_same_length([
        base_positive_prompt_embeds, base_negative_prompt_embeds
    ])

    prompt_params = {
        "prompt_embeds": base_positive_prompt_embeds,
        "pooled_prompt_embeds": base_positive_prompt_pooled,
        "negative_prompt_embeds": base_negative_prompt_embeds,
        "negative_pooled_prompt_embeds": base_negative_prompt_pooled,
    }

    return prompt_params


def generate_image(args, pipe, refiner, prompt_params, output_dir):
    use_refiner = args.refiner
    device = args.device

    gen_steps = args.samples
    ref_steps = args.refinesamples
    seed = args.seed

    if seed == -1:
        seed = random.randint(0, sys.maxsize)

    pipeline_params = {
        **prompt_params,
        "output_type": "latent" if use_refiner else "pil",
        "generator": torch.Generator(device).manual_seed(seed),
        "num_inference_steps": gen_steps,
    }

    print("Generating image")
    images = pipe(**pipeline_params).images

    if use_refiner:
        refiner_params = {
            **prompt_params,
            "image": images,
            "generator": torch.Generator(device).manual_seed(seed),
            "num_inference_steps": ref_steps,
        }

        print("Refining image")
        images = refiner(**refiner_params).images


    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    image_path = os.path.join(output_dir, f"output_{timestamp}_{seed}.jpg")
    images[0].save(image_path)

    print(f"Seed:\t{seed}\tImage saved at: {image_path}")


def main():
    parser = argparse.ArgumentParser(description='XL Stable-Diffusion Generation')

    parser.add_argument('-d', '--device', type=str, help='Device to use for AI', default='cpu')
    parser.add_argument('-r', '--refiner', action='store_true', help='Switch to use refiner')
    parser.add_argument('-p', '--prompt', type=str, help='Prompt for image creation', default="beautiful lady, (freckles), big smile, blue eyes, long dark hair, dark smokey makeup, hyperdetailed photography, soft light, head and shoulders portrait, cover")
    parser.add_argument('-n', '--negprompt', type=str, help='Negative rompt for image creation', default="(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)")
    parser.add_argument('-s', '--samples', type=int, help='Number of samples', default=30)
    parser.add_argument('-S', '--seed', type=int, help='Seed Value', default=-1)
    parser.add_argument('-R', '--refinesamples', type=int, help='Number refiner of samples', default=20)
    parser.add_argument('-o', '--output', type=str, help='Output Directory', default='output_images')
    parser.add_argument('-i', '--interations', type=int, help='Number of image iterations', default=1)

    args = parser.parse_args()

    # Output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    pipe, refiner = setup_pipe(args)
    prompt_params = process_prompts(pipe, args.prompt, args.negprompt)
    print(f"Prompt: {args.prompt}\nNegative Prompt: {args.negprompt}")
    for _ in range(args.interations):
        generate_image(args, pipe, refiner, prompt_params, output_dir)


if __name__ == '__main__':
    main()