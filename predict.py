# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

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
    ) -> Path:
        """Run a single prediction on the model"""
        os.system(f"animatediff generate -c {config_path} -W {width} -H {height} -L {length} -C {context}")
        return Path(config_path)
