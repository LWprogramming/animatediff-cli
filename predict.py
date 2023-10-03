# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import textwrap
from typing import List
import uuid
import cog
import os
import sys

# from pathlib import Path
import pathlib  # do NOT import Path because it will conflict with cog.Path

sys.path.insert(0, "src")  # make animatediff importable
from animatediff.cli import generate


class Predictor(cog.BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    def predict(
        self,
        config_path: str = cog.Input(
            description="Path to the configuration file",
            default="config/prompts/prompt_travel.json",
        ),
        width: int = cog.Input(description="Width of the output", default=256),
        height: int = cog.Input(description="Height of the output", default=384),
        length: int = cog.Input(description="Number of frames to generate", default=128),
        context: int = cog.Input(description="Number of context frames to use", default=16),
        out_dir: str = cog.Input(description="Directory for output folders", default="output"),
        no_frames: bool = cog.Input(description="Don't save frames, only the animation", default=True),
        save_merged: bool = cog.Input(description="Save a merged animation of all prompts", default=True),
        head_prompt: str = cog.Input(
            description="OPTIONAL: Override the head prompt from the model config, which is added as a suffix to each prompt. Here's a possible head prompt: robot, best quality, masterpiece, futuristic design",
            default=None,
        ),
        tail_prompt: str = cog.Input(
            description="OPTIONAL: Override the tail prompt from the model config, which is added as a suffix to each prompt. Here's a possible tail prompt: forward facing, standing, full body",
            default=None,
        ),
        prompt_map: str = cog.Input(
            description=textwrap.dedent(
                """
            OPTIONAL: Override the prompt map from the model config. Format: 'number: description' per line. Here's a possible prompt map (remember to insert newlines between each line):
            0: walking through a dense jungle scene. vines, trees, plants. camouflage and green metal finish
            32: walking slowly across the surface of mars. red dirt, rocks, mars landscape. bright metallic finish reflecting surroundings
            64: exploring an underwater shipwreck. fish, corals, sea plants, bubbles. rusted metal finish and ocean plant growth on body
            96: crossing a bridge over a busy futuristic city with flying cars. neon lights, glass buildings, bright colors. clean white and silver metal finish"""
            ),
            default=None,
        ),
        n_prompt: str = cog.Input(
            description=textwrap.dedent(
                """OPTIONAL: Override the negative prompt from the model config. Single string input-- haven't implemented multiple negative prompts.
            Unfortunately, to make things optional in Replicate, I have to make the default none so I can't add a suggestion directly in the box. Here's a possible negative prompt: worst quality, low quality, cropped, lowres, text, jpeg artifacts, multiple view, nude, nsfw, drawn incorrectly"""
            ),
            default=None,
        ),
        seed: int = cog.Input(
            description="OPTIONAL: Seed for the random number generator",
            default=None,
        ),
    ) -> List[cog.Path]:
        """Run a single prediction on the model"""
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # Parse prompt_map and n_prompt
        if prompt_map is not None:
            # The prompt_map is expected to be a string with each line in the format 'number: description'
            # This block of code converts that string into a dictionary for easier access later on.
            # For example, if prompt_map is "1: cat\n2: dog", it will be converted to {1: "cat", 2: "dog"}
            # line is not "" means we ignore empty lines and don't accidentally mess up there.
            try:
                prompt_map = {
                    int(line.split(":")[0]): line.split(":")[1].strip()
                    for line in prompt_map.strip().split("\n")
                    if line is not ""
                }
            except Exception as e:
                raise ValueError(
                    "prompt_map is not formatted correctly. It should be 'number: description' per line"
                ) from e

        print("beginning generation")
        generate(
            model_name_or_path="stable-diffusion-v1-5",  # just loads from data/models/huggingface/stable-diffusion-v1-5, where we have the weights loaded locally, so we don't need to download them
            config_path=pathlib.Path(config_path),
            width=width,
            height=height,
            length=length,
            context=context,
            out_dir=pathlib.Path(out_dir),
            no_frames=no_frames,
            save_merged=save_merged,
            head_prompt=head_prompt,
            tail_prompt=tail_prompt,
            prompt_map=prompt_map,
            n_prompt=n_prompt,
            seed=seed,
        )
        print("generation complete")

        # return [cog.Path(os.path.join(out_dir, file)) for file in os.listdir(out_dir)]
        def get_files_recursive(path_str):
            for root, dirs, files in os.walk(path_str):
                for file in files:
                    yield cog.Path(pathlib.Path(root) / file)  # ugh the typing here lol

        return list(get_files_recursive(out_dir))
