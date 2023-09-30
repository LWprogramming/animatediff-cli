from typing import Optional, List
from pathlib import Path

from animatediff import __version__, console, get_dir
from animatediff.generate import create_pipeline, run_inference
from animatediff.settings import (
    InferenceConfig,
    ModelConfig,
    get_infer_config,
    get_model_config,
)
from animatediff.utils.model import get_base_model
from animatediff.utils.pipeline import send_to_device
from animatediff.utils.util import save_frames, save_video


from cog import BasePredictor, Input
from datetime import datetime
import torch


class Predictor(BasePredictor):
    def setup(self):
        # Load the model into memory to make running multiple predictions efficient
        device = "cuda"

        # Get the base model if we don't have it already
        self.base_model_path = get_base_model(
            "runwayml/stable-diffusion-v1-5", local_dir=get_dir("data/models/huggingface")
        )

        # Create the pipeline
        # TODO: right now the config file is hardcoded and we need to change that later
        # we also totally ignore the prompts in the config file itself since we just set it in predict()
        use_xformers = False  # TODO: Look into this later", default=False),
        force_half_vae = False  # TODO: nput(description="Look into this later", default=False),

        print("creating pipeline...")
        self.pipeline = create_pipeline(
            base_model=self.base_model_path,
            model_config=get_model_config("01-ToonYou.json"),
            infer_config=get_infer_config(),
            use_xformers=use_xformers,
        )

        print("sending pipeline to device...")
        # Send the pipeline to device
        device: torch.device = torch.device(device)
        self.pipeline = send_to_device(
            self.pipeline, device, freeze=True, force_half=force_half_vae, compile=True
        )

    def predict(
        self,
        prompt: str = Input(
            description="List of prompts separated by newlines. Naturally each individual prompt itself cannot contain newlines.",
            default="",
        ),
        n_prompt: str = Input(description="Negative prompt to be used for all stages for now", default=""),
        seed: int = Input(description="Seed for random number generator", default=42),
        steps: int = Input(description="Number of steps for the inference", default=20),
        guidance_scale: float = Input(description="Guidance scale for the inference", default=7.5),
        width: int = Input(description="Width of the image", default=576),
        height: int = Input(description="Height of the image", default=576),
        duration: int = Input(description="Duration in frames", default=16),
        context: int = Input(
            description="Context for the image generation. Suggest 8 for low memory, 16-20 as default, 24 as the max",
            default=8,
        ),
        no_frames: bool = Input(description="Initialize no_frames", default=False),
        save_merged: bool = Input(description="Initialize save_merged", default=True),
    ) -> List[Path]:
        save_dir = f"animatediff_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_dir.mkdir(parents=True, exist_ok=True)
        prompts = prompt.split("\n")
        # TODO: figure out this later iguess
        overlap = None
        stride = None
        # Run inference and save frames
        output_paths = []

        for i in range(duration):
            print(f"Generating frame {i} of {duration}")
            output = run_inference(
                pipeline=self.pipeline,
                prompt=prompts[i],
                n_prompt=n_prompt,
                seed=seed,
                steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                duration=duration,
                idx=i,
                out_dir=save_dir,
                context_frames=context,
                context_overlap=overlap,
                context_stride=stride,
                clip_skip=1,
            )
            output_paths.append(output)
            torch.cuda.empty_cache()
            if not no_frames:
                save_frames(output, save_dir.joinpath(f"{i}-{seed}"))

        # Save a merged animation of all prompts
        if save_merged:
            video_path = save_dir.joinpath("final.gif")
            save_video(output, video_path)
            output_paths.append(video_path)

        return output_paths
