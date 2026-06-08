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
    Adapt an infer-style config for labeled evaluation.

    Image list comes from ``path_to_all_benchmarkset_txt`` (mapped to
    ``path_to_all_txt`` by ad_values_nececeary_for_dataset_loader_creation).
    """
    ad_values_nececeary_for_dataset_loader_creation(cfg)
    # path_to_all_txt == path_to_all_benchmarkset_txt; sdfi_dataset reads that file.
    # Empty valid list => ListSplitter puts every image in the train split.
    # (Assigning all names to valid.txt leaves train empty and breaks fastai setup.)
    empty_valid = cfg["path_to_all_benchmarkset_txt"].parent / ".eval_empty_valid.txt"
    empty_valid.write_text("")
    cfg["path_to_valid_txt"] = empty_valid
    if "ignore_index" not in cfg:
        cfg["ignore_index"] = 0
    if cfg.get("dev_mode"):
        print("eval: dev_mode disabled so the full benchmark set is evaluated")
    cfg["dev_mode"] = False
    return cfg


def center_crop_hw(tensor, crop_size):
    """Center-crop the last two spatial dimensions (H, W)."""
    h, w = tensor.shape[-2], tensor.shape[-1]
    crop_size = int(crop_size)
    y0 = int((h - crop_size) / 2)
    x0 = int((w - crop_size) / 2)
    y1 = y0 + crop_size
    x1 = x0 + crop_size
    return tensor[..., y0:y1, x0:x1]


def evaluate_pixel_accuracy(learn, dl, ignore_index, crop_size=False):
    """
    Run inference on dl and return global pixel accuracy over all batches.
    """
    learn.model.eval()
    correct = 0
    total = 0
    ignore_index = int(ignore_index)

    with torch.no_grad():
        for batch in dl:
            inp, targ = batch
            preds = learn.model(inp)
            if crop_size:
                preds = center_crop_hw(preds, crop_size)
                targ = center_crop_hw(targ, crop_size)
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
    eval_dl = dls.train
    n_images = len(eval_dl.items)
    print(
        f"Evaluating {n_images} images listed in path_to_all_benchmarkset_txt "
        f"({cfg['path_to_all_benchmarkset_txt']}), resolved under path_to_images "
        f"({cfg['path_to_images']}), im_type={cfg['im_type']}"
    )

    trainer = train.BasicTrainingFastai2(cfg, dls)
    model_path = pathlib.Path(cfg["model_to_load"]).resolve()
    trainer.learn.load(str(model_path).rstrip(".pth"), weights_only=False)

    if torch.cuda.is_available():
        trainer.learn.model.cuda()

    crop_size = cfg.get("crop_size") and int(cfg["crop_size"])
    if crop_size:
        print(f"Using center crop {crop_size}x{crop_size} (same as infer.py)")

    accuracy = evaluate_pixel_accuracy(
        trainer.learn,
        eval_dl,
        cfg["ignore_index"],
        crop_size=crop_size,
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
