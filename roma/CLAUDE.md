# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ROMA (Region Similarity Matching) is a PyTorch research codebase for unpaired nighttime infrared to daytime visible translation, built on top of the CycleGAN/Pix2Pix and CUT codebases. The main workflows in this repository are training (`train.py`) and offline inference (`test.py`); there is no separate package build step.

## Common Commands

### Training

Video mode uses `models/roma_model.py` with `data/unaligned_double_dataset.py`. Each training sample is a single 512x256 image made of two adjacent 256x256 frames concatenated along width.

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --dataroot /path/to/dataset \
  --name ROMA_name \
  --dataset_mode unaligned_double \
  --no_flip \
  --local_nums 64 \
  --display_env ROMA_env \
  --model roma \
  --side_length 7 \
  --lambda_spatial 5.0 \
  --lambda_global 5.0 \
  --lambda_motion 1.0 \
  --atten_layers 1,3,5 \
  --lr 0.00001
```

Image mode uses the same ROMA training model with the standard unaligned image dataset.

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --dataroot /path/to/dataset \
  --name ROMA_name \
  --dataset_mode unaligned \
  --local_nums 64 \
  --display_env ROMA_env \
  --model roma \
  --side_length 7 \
  --lambda_spatial 5.0 \
  --lambda_global 5.0 \
  --atten_layers 1,3,5 \
  --lr 0.00001
```

The checked-in script examples are `scripts/train.sh` and `scripts/test.sh`.

### Inference

```bash
CUDA_VISIBLE_DEVICES=0 python3 test.py \
  --dataroot /path/to/test_dataset \
  --checkpoints_dir ./checkpoints \
  --name experiment_name \
  --model roma_single \
  --num_test 10000 \
  --epoch latest
```

Inference writes HTML and images under `./results/<name>/<phase>_<epoch>/`.

### Validation / sanity checks

There is no checked-in lint, formatter, or unit-test configuration in this repository. The lightweight validation command that works without datasets is a Python syntax check:

```bash
python3 -m py_compile train.py test.py options/*.py data/*.py models/*.py util/*.py
```

If you need to inspect all resolved flags for a run, use the entrypoint help after installing the PyTorch dependencies:

```bash
python3 train.py --help
python3 test.py --help
```

## Dataset Layout

Expected dataset layout is driven by `opt.phase`:

```text
dataset_root/
  trainA/
  trainB/
  testA/
  testB/
```

If `testA`/`testB` do not exist, both `unaligned` and `unaligned_double` datasets fall back to `valA`/`valB` during test mode.

For `unaligned_double`, each image is split into `(A0, A1)` or `(B0, B1)` by cropping left and right 256x256 halves in `data/unaligned_double_dataset.py`. The same random crop/transform is then applied to both frames and both domains so temporal alignment is preserved.

## Architecture

### Execution flow

- `train.py` parses options, creates the dataset through `data.create_dataset(opt)`, creates the model through `models.create_model(opt)`, and runs the epoch/iteration loop.
- `test.py` uses the same dynamic option system, forces single-threaded batch-1 evaluation, loads the model, and saves results as an HTML gallery.
- `options/base_options.py` is the root of the CLI system. Parsing order is: base options → train/test options → model-specific `modify_commandline_options` → dataset-specific `modify_commandline_options`.
- Parsed options are also written to `checkpoints/<experiment>/<phase>_opt.txt`, so changing defaults affects saved experiment metadata.

### Dynamic model and dataset registries

This codebase relies on name-based lookup rather than a static registry:

- `models/__init__.py` resolves `--model roma` to `models/roma_model.py` and `ROMAModel`.
- `data/__init__.py` resolves `--dataset_mode unaligned_double` to `data/unaligned_double_dataset.py` and `UnalignedDoubleDataset`.

When adding a new model or dataset, the filename and class name must follow the existing `[name]_model.py` / `[name]_dataset.py` convention or the loader will fail.

### ROMA-specific model design

The main ROMA logic lives in `models/roma_model.py`:

- `netG` is the image translation generator built through `models/networks.py`.
- `netD_ViT` is an `MLPDiscriminator` that operates on token features rather than raw pixels.
- `netPreViT` is a frozen pretrained `vit_base_patch16_384` loaded from the vendored `timm/` copy and used only as a feature extractor.
- Training generates `fake_B0` and `fake_B1` from adjacent infrared frames, extracts multi-layer ViT tokens for real and fake images, and combines adversarial loss with cross-domain structural losses.

Important losses and switches:

- `--lambda_GAN`: adversarial loss on ViT-token features.
- `--lambda_global`: global structural consistency.
- `--lambda_spatial`: local structural consistency using sampled token neighborhoods.
- `--lambda_motion`: temporal consistency between adjacent frames; only meaningful for the video path.
- `--atten_layers`: selects which ViT block outputs participate in similarity computation.
- `--which_D_layer`: selects which extracted ViT layer is routed into the MLP discriminator.

`models/roma_single_model.py` is the single-frame variant used for image-style input and inference. It keeps the same frozen-ViT idea but operates on one frame instead of adjacent pairs and does not use the motion loss path.

### Shared framework pieces

- `models/base_model.py` owns checkpoint loading/saving, scheduler setup, DataParallel wrapping, and the train/test interface expected by all models.
- `models/networks.py` contains generator/discriminator builders, GAN losses, schedulers, and the `MLPDiscriminator` used by ROMA.
- `util/visualizer.py` handles both Visdom live displays and HTML snapshot export. Training can create `checkpoints/<name>/web/` and `loss_log.txt`; test-time export goes through `util/html.py` into `results/`.

### Vendored dependencies

`timm/` is vendored into the repository and is part of the runtime, not just a third-party snapshot. Changes there can alter the frozen ViT feature extraction path used by ROMA.
