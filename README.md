# animatediff
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/neggles/animatediff-cli/main.svg)](https://results.pre-commit.ci/latest/github/neggles/animatediff-cli/main)

animatediff refactor, ~~because I can.~~ with significantly lower VRAM usage.

Also, **infinite generation length support!** yay!

# LoRA loading is ABSOLUTELY NOT IMPLEMENTED YET!

PRs welcome! 😆😅

This can theoretically run on CPU, but it's not recommended. Should work fine on a GPU, nVidia or otherwise,
but I haven't tested on non-CUDA hardware. Uses PyTorch 2.0 Scaled-Dot-Product Attention (aka builtin xformers)
by default, but you can pass `--xformers` to force using xformers if you *really* want.

## How to use

I should write some more detailed steps, but here's the gist of it:

```sh
git clone https://github.com/neggles/animatediff-cli
cd animatediff-cli
python3.10 -m venv .venv
source .venv/bin/activate
# install Torch. Use whatever your favourite torch version >= 2.0.0 is, but, good luck on non-nVidia...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# install the rest of all the things (probably! I may have missed some deps.)
python -m pip install -e '.[dev]'
# you should now be able to
animatediff --help
# There's a nice pretty help screen with a bunch of info that'll print here.
```

From here you'll need to put whatever checkpoint you want to use into `data/models/sd`, copy
one of the prompt configs in `config/prompts`, edit it with your choices of prompt and model (model
paths in prompt .json files are **relative to `data/`**, e.g. `models/sd/vanilla.safetensors`), and
off you go.

Then it's something like (for an 8GB card):
```sh
animatediff generate -c 'config/prompts/waifu.json' -W 576 -H 576 -L 128 -C 16
```
You may have to drop `-C` down to 8 on cards with less than 8GB VRAM, and you can raise it to 20-24
on cards with more. 24 is max.

N.B. generating 128 frames is _**slow...**_

## RiFE!

I have added experimental support for [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan)
using the `animatediff rife interpolate` command. It has fairly self-explanatory help, and it has
been tested on Linux, but I've **no idea** if it'll work on Windows.

Either way, you'll need ffmpeg installed on your system and present in PATH, and you'll need to
download the rife-ncnn-vulkan release for your OS of choice from the GitHub repo (above). Unzip it, and
place the extracted folder at `data/rife/`. You should have a `data/rife/rife-ncnn-vulkan` executable, or `data\rife\rife-ncnn-vulkan.exe` on Windows.

You'll also need to reinstall the repo/package with:
```py
python -m pip install -e '.[rife]'
```
or just install `ffmpeg-python` manually yourself.

Default is to multiply each frame by 8, turning an 8fps animation into a 64fps one, then encode
that to a 60fps WebM. (If you pick GIF mode, it'll be 50fps, because GIFs are cursed and encode
frame durations as 1/100ths of a second).

Seems to work pretty well...

## TODO:

In no particular order:

- [x] Infinite generation length support
- [x] RIFE support for motion interpolation (`rife-ncnn-vulkan` isn't the greatest implementation)
- [x] Export RIFE interpolated frames to a video file (webm, mp4, animated webp, hevc mp4, gif, etc.)
- [x] Generate infinite length animations on a 6-8GB card (at 512x512 with 8-frame context, but hey it'll do)
- [x] Torch SDP Attention (makes xformers optional)
- [x] Support for `clip_skip` in prompt config
- [x] Experimental support for `torch.compile()` (upstream Diffusers bugs slow this down a little but it's still zippy)
- [x] Batch your generations with `--repeat`! (e.g. `--repeat 10` will repeat all your prompts 10 times)
- [x] Call the `animatediff.cli.generate()` function from another Python program without reloading the model every time
- [x] Drag remaining old Diffusers code up to latest (mostly)
- [ ] Add a webUI (maybe, there are people wrapping this already so maybe not?)
- [ ] img2img support (start from an existing image and continue)
- [ ] Stop using custom modules where possible (should be able to use Diffusers for almost all of it)
- [ ] Automatic generate-then-interpolate-with-RIFE mode

## Credits:

see [guoyww/AnimateDiff](https://github.com/guoyww/AnimateDiff) (very little of this is my work)

n.b. the copyright notice in `COPYING` is missing the original authors' names, solely because
the original repo (as of this writing) has no name attached to the license. I have, however,
used the same license they did (Apache 2.0).

# GETTING THINGS SET UP ON REPLICATE
- install things using the writeup above. you'll have to get torch with cuda118 on linux bc no such thing exists on mac of course

```
# cd into src/animatediff because imports and that sort of stuff is cog being silly
# then get the cog executable
# TODO: this curl command can fail because the release names are like
# https://github.com/replicate/cog/releases/download/v0.8.6/cog_linux_x86_64
# which doesn't match the curl command below but they're INCONSISTENTLY mismatched which sucks
curl -o ./cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
chmod +x ./cog
echo 'export PATH="$HOME/cog:$PATH"' >> ~/.bashrc
./cog init

echo "REMEMBER THE cog.yaml TO INCLUDE GPU IN BUILD!!!"
# also add cuda: "11.8" so that you can pull out the right torch version bc that's what we need here.
# and you're going to need to freeze requirements.txt and add `--extra-index-url https://download.pytorch.org/whl/cu118` to the top of requirements.txt, see here https://github.com/replicate/cog/issues/1266#issuecomment-1741832134 and discussion below
# also fix the animatediff install so that you install from repo via https like -e git+https://github.com/LWprogramming/animatediff-cli.git@8a605f73d9cbee986855d5e9519c7e3d3bb48392#egg=animatediff (you will probably want to change this to the original repo or your personal fork in case I end up deleting this fork at some point or whatever). Cog fails if you try to download via ssh so if you git cloned via ssh or something you'll need to manually fix that.
# and also make sure your .dockerignore is manually ignoring stuff like venv files
echo "And run ./cog login if you haven't already"
echo "And also change the cog yaml to whatever prediction file you want to use

# Push trained model to Replicate
# Assumes you've already run
# ./cog login
# and created a project named PROJECT_NAME

./cog push r8.im/lwprogramming/${PROJECT_NAME} # uses predict.py by default-- probably should change to replicate_predict.py in the yaml

```