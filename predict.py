# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from typing import List
import uuid
from cog import BasePredictor, Input, Path
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")  # make animatediff importable
from animatediff.cli import generate


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    def predict(
        self,
        config_path: str = Input(
            description="Path to the configuration file",
            default="config/prompts/prompt_travel.json",
        ),
        width: int = Input(description="Width of the output", default=256),
        height: int = Input(description="Height of the output", default=384),
        length: int = Input(description="Number of frames to generate", default=128),
        context: int = Input(description="Number of context frames to use", default=16),
        out_dir: str = Input(description="Directory for output folders", default=str(uuid.uuid4())),
        no_frames: bool = Input(description="Don't save frames, only the animation", default=True),
        save_merged: bool = Input(description="Save a merged animation of all prompts", default=True),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if os.path.exists(out_dir):
            raise FileExistsError(f"Output directory {out_dir} already exists.")
        os.makedirs(out_dir)
        generate(
            model_name_or_path="stable-diffusion-v1-5",  # just loads from data/models/huggingface/stable-diffusion-v1-5, where we have the weights loaded locally, so we don't need to download them
            config_path=Path(config_path),
            width=width,
            height=height,
            length=length,
            context=context,
            out_dir=Path(out_dir),
            no_frames=no_frames,
            save_merged=save_merged,
        )

        # return [Path(os.path.join(out_dir, file)) for file in os.listdir(out_dir)]
        def get_files_recursive(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    yield Path(root) / file

        return list(get_files_recursive(out_dir))
