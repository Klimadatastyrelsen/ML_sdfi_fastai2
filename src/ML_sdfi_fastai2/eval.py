#!/usr/bin/env python
"""
Evaluate a trained segmentation model on a labeled benchmark set.

Computes pixel classification accuracy (matching training valid_accuracy):
fraction of non-ignored label pixels predicted correctly.

Which images are evaluated
--------------------------
All images listed in the config key ``path_to_all_benchmarkset_txt`` (for the
example infer config this is ``.../data/all.txt``). Each non-empty line is a
path relative to ``path_to_images`` (e.g. ``rgb/some_tile.tif``). Lines that do
not contain ``im_type`` (typically ``.tif``) are skipped. For each image, the
matching label is loaded from ``path_to_labels`` using the same filename stem.
This is the same image list ``infer.py`` uses for benchmark inference.
"""

import argparse
import pathlib

import torch

import ML_sdfi_fastai2.sdfi_dataset as sdfi_dataset
import ML_sdfi_fastai2.train as train
import ML_sdfi_fastai2.utils.utils as sdfi_utils
from ML_sdfi_fastai2.infer import ad_values_nececeary_for_dataset_loader_creation


def prepare_eval_config(cfg):
    """
    Adapt a training-style config so the benchmark list is the validation split.

    ``path_to_all_txt`` must stay the full training list. Putting every benchmark
    name into the valid split and leaving train empty breaks fastai. Train-only
    augmentations (split_idx=0) must not run; ``dls.valid`` applies only the
    validation transforms (centre crop) plus Normalize, which is what training
    logs as pixel_accuracy.
    """
    all_txt = cfg.get("path_to_all_txt")
    ad_values_nececeary_for_dataset_loader_creation(cfg)
    if all_txt:
        cfg["path_to_all_txt"] = all_txt
    cfg["path_to_valid_txt"] = cfg["path_to_all_benchmarkset_txt"]
    if "ignore_index" not in cfg:
        cfg["ignore_index"] = 0
    if cfg.get("dev_mode"):
        print("eval: dev_mode disabled so the full benchmark set is evaluated")
    cfg["dev_mode"] = False
    return cfg


def evaluate_pixel_accuracy(learn, dl, ignore_index, use_fp16=False):
    """
    Run inference on dl and return global pixel accuracy over all batches.

    The dataloader is expected to already match training validation (centre crop
    on the input, no train augmentations). Do not crop the prediction afterwards:
    that would score a different forward pass than the one that was logged.
    """
    learn.model.eval()
    correct = 0
    total = 0
    ignore_index = int(ignore_index)
    device = next(learn.model.parameters()).device

    with torch.no_grad():
        for batch in dl:
            inp, targ = batch
            inp = inp.to(device)
            targ = targ.to(device)
            if use_fp16 and device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    preds = learn.model(inp)
                preds = preds.float()
            else:
                preds = learn.model(inp)
            labels = targ.squeeze(1)
            mask = labels != ignore_index
            if mask.sum() == 0:
                continue
            pred_classes = preds.argmax(1)
            correct += (pred_classes[mask] == labels[mask]).sum().item()
            total += mask.sum().item()

    if total == 0:
        return 0.0
    return correct / total


def run_eval(config_path):
    cfg = sdfi_utils.load_settings_from_config_file(config_path)
    prepare_eval_config(cfg)

    print("##########################################")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"PyTorch is using GPU: {device_name}")
    else:
        print("PyTorch is using CPU")
    print("##########################################")

    dls = sdfi_dataset.get_dataset(cfg)
    eval_dl = dls.valid
    n_images = len(eval_dl.items)
    print(
        f"Evaluating {n_images} images listed in path_to_all_benchmarkset_txt "
        f"({cfg['path_to_all_benchmarkset_txt']}), resolved under path_to_images "
        f"({cfg['path_to_images']}), im_type={cfg['im_type']}"
    )

    trainer = train.BasicTrainingFastai2(cfg, dls)
    model_path = pathlib.Path(cfg["model_to_load"]).resolve()
    trainer.learn.load(str(model_path.with_suffix("")), weights_only=False)

    if torch.cuda.is_available():
        trainer.learn.model.cuda()

    if "crop" in (cfg.get("transforms") or []):
        print(f"Validation centre crop {cfg.get('crop_size')} is applied by the dataloader, same as training")

    accuracy = evaluate_pixel_accuracy(
        trainer.learn,
        eval_dl,
        cfg["ignore_index"],
        use_fp16=bool(cfg.get("to_fp16")),
    )

    print(f"Images evaluated: {n_images}")
    print(f"ignore_index: {cfg['ignore_index']}")
    print(f"Pixel accuracy: {accuracy:.6f} ({accuracy * 100:.2f}%)")
    return accuracy


if __name__ == "__main__":
    usage_example = (
        "Example usage:\n"
        "python src/ML_sdfi_fastai2/eval.py\n"
        "python src/ML_sdfi_fastai2/eval.py --config configs/example_configs/infer_example_dataset.ini\n"
        "python src/ML_sdfi_fastai2/eval.py --config config_a.ini config_b.ini\n"
    )
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation model pixel accuracy on a labeled benchmark set.",
        epilog=usage_example,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="+",
        default=["configs/example_configs/infer_example_dataset.ini"],
        help="One or more paths to infer-style experiment config files",
    )
    args = parser.parse_args()
    for config_path in args.config:
        run_eval(config_path)
