# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from typing import List
import uuid
from cog import BasePredictor, Input, Path
import os
import sys


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
        os.system(
            f"animatediff generate --config-path {config_path} --width {width} --height {height} --length {length} --context {context} --out-dir {out_dir} --no-frames {no_frames} --save-merged {save_merged}"
        )
        return [Path(os.path.join(out_dir, file)) for file in os.listdir(out_dir)]
