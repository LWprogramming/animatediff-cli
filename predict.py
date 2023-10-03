# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

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
        out_dir: str = cog.Input(description="Directory for output folders", default=str(uuid.uuid4())),
        no_frames: bool = cog.Input(description="Don't save frames, only the animation", default=True),
        save_merged: bool = cog.Input(description="Save a merged animation of all prompts", default=True),
    ) -> List[cog.Path]:
        """Run a single prediction on the model"""
        if os.path.exists(out_dir):
            raise FileExistsError(f"Output directory {out_dir} already exists.")
        os.makedirs(out_dir)
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
        )

        # return [cog.Path(os.path.join(out_dir, file)) for file in os.listdir(out_dir)]
        def get_files_recursive(path_str):
            for root, dirs, files in os.walk(path_str):
                for file in files:
                    yield cog.Path(pathlib.Path(root) / file)  # ugh the typing here lol

        return list(get_files_recursive(out_dir))
