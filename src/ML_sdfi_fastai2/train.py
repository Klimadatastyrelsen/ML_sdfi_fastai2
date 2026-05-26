#!/usr/bin/env python
# coding: utf-8
"""
Train a model to do semantic segmentation

Supported architectures:
- UNet (fastai)
- timm UNet (EfficientNet / ConvNeXt)
- SegFormer
- Swin + UPerNet
- ConvNeXt V2 + UPerNet
"""

import os
import sys
import time
import json
import random
import pathlib
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fastai.basics import *
from fastai.vision.all import *
from fastai.callback.all import *
from fastai.vision.learner import unet_learner
from fastai.callback.schedule import minimum, steep, slide, valley
from fastai.vision.all import GradientAccumulation

import utils.utils as sdfi_utils
import sdfi_dataset

from wwf.vision.timm import timm_unet_learner


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------
def make_deterministic():
    print("Enabling deterministic training")
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# CSV logger with LR
# ---------------------------------------------------------------------
class CSVLoggerWithLR(CSVLogger):
    def after_epoch(self):
        lrs = [g['lr'] for g in self.learn.opt.param_groups]
        log_values = self.learn.recorder.log + lrs

        if not hasattr(self, 'header_written'):
            self.file.write(','.join(self.learn.recorder.metric_names +
                                     [f'lr_{i}' for i in range(len(lrs))]) + '\n')
            self.header_written = True

        self.file.write(','.join(map(str, log_values)) + '\n')
        self.file.flush()


# ---------------------------------------------------------------------
# Batch checkpoint callback
# ---------------------------------------------------------------------
class DoThingsAfterBatch(Callback):
    """Save model after n batches"""
    def __init__(self, n_batch: int = 200_000):
        self.iter_string = "batch_string_NOT_set"
        self._modulus_faktor = n_batch

    def after_batch(self):
        if self._modulus_faktor < 2:
            return
        if self.iter % self._modulus_faktor == (self._modulus_faktor - 1):
            print(f"Iter: {self.iter} of {self.n_iter}")
            self.iter_string = f"Batch_model_{self.epoch}_{self.iter}"
            x_cpu = self.loss.cpu()
            self.lr_string = f"  loss={x_cpu.detach().numpy()}"
            print("Batch save filename:" + self.iter_string + self.lr_string)
            self.learn.save(self.iter_string)
            print("Batch model saved!")


# ---------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target = target.squeeze(1)
        mask = target != self.ignore_index

        num_classes = pred.shape[1]
        target_oh = F.one_hot(target[mask], num_classes).permute(1, 0).float()
        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, num_classes)[mask.reshape(-1)]

        intersection = (pred_flat * target_oh).sum(0)
        union = pred_flat.sum(0) + target_oh.sum(0)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        target = target.squeeze(1)
        ce = F.cross_entropy(pred, target.long(), reduction='none',
                             ignore_index=self.ignore_index)
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5, ignore_index=255, class_weights=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target.squeeze(1).long())
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


# ---------------------------------------------------------------------
# SegFormer wrapper
# ---------------------------------------------------------------------
class SegFormerWrapper(nn.Module):
    """Wrapper for SegFormer models from transformers library"""
    def __init__(self, model_name, num_classes, n_in=3, pretrained=True, ignore_index=255):
        super().__init__()
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerConfig
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            if n_in != 3:
                self._adapt_input_channels(n_in)
        else:
            config = SegformerConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            self.model = SegformerForSemanticSegmentation(config)
            if n_in != 3:
                self._adapt_input_channels(n_in)
    
    def _adapt_input_channels(self, n_in):
        segformer = self.model.segformer
        if hasattr(segformer, "encoder"):
            old_conv = segformer.encoder.patch_embeddings[0].proj
        else:
            old_conv = segformer.stages[0].patch_embeddings.proj
        new_conv = nn.Conv2d(
            n_in, 
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        if new_conv.bias is not None:
            nn.init.constant_(new_conv.bias, 0)
        if n_in >= 3 and old_conv.weight.shape[1] == 3:
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
        if hasattr(segformer, "encoder"):
            segformer.encoder.patch_embeddings[0].proj = new_conv
        else:
            segformer.stages[0].patch_embeddings.proj = new_conv
    
    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        return logits


# ---------------------------------------------------------------------
# Swin + UPerNet wrapper
# ---------------------------------------------------------------------
class SwinUPerNetWrapper(nn.Module):
    """Wrapper for Swin Transformer + UPerNet from transformers"""
    def __init__(self, model_name, num_classes, n_in=3, pretrained=True, ignore_index=255):
        super().__init__()
        try:
            from transformers import AutoModelForSemanticSegmentation, UperNetConfig
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        if pretrained:
            self.model = AutoModelForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            if n_in != 3:
                self._adapt_input_channels(n_in)
        else:
            config = UperNetConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            self.model = AutoModelForSemanticSegmentation(config)
            if n_in != 3:
                self._adapt_input_channels(n_in)
    
    def _adapt_input_channels(self, n_in):
        try:
            if hasattr(self.model, 'backbone'):
                old_patch_embed = self.model.backbone.embeddings.patch_embeddings.projection
            elif hasattr(self.model, 'swin'):
                old_patch_embed = self.model.swin.embeddings.patch_embeddings.projection
            else:
                print("Warning: Could not find patch embedding layer to adapt")
                return
            
            new_patch_embed = nn.Conv2d(
                n_in,
                old_patch_embed.out_channels,
                kernel_size=old_patch_embed.kernel_size,
                stride=old_patch_embed.stride,
                padding=old_patch_embed.padding,
                bias=old_patch_embed.bias is not None
            )
            
            nn.init.kaiming_normal_(new_patch_embed.weight, mode='fan_out', nonlinearity='relu')
            if new_patch_embed.bias is not None:
                nn.init.constant_(new_patch_embed.bias, 0)
            
            if n_in >= 3 and old_patch_embed.weight.shape[1] == 3:
                with torch.no_grad():
                    new_patch_embed.weight[:, :3] = old_patch_embed.weight
            
            if hasattr(self.model, 'backbone'):
                self.model.backbone.embeddings.patch_embeddings.projection = new_patch_embed
            elif hasattr(self.model, 'swin'):
                self.model.swin.embeddings.patch_embeddings.projection = new_patch_embed
        except Exception as e:
            print(f"Warning: Could not adapt input channels: {e}")
    
    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        return logits


# ---------------------------------------------------------------------
# ConvNeXt V2 + UPerNet (transformers-based)
# ---------------------------------------------------------------------
class ConvNeXtV2UPerNetWrapper(nn.Module):
    """ConvNeXt V2 backbone + UPerNet decoder using transformers library"""
    def __init__(self, backbone_name, num_classes, n_in, pretrained=True):
        super().__init__()
        try:
            from transformers import AutoModelForSemanticSegmentation, UperNetConfig
        except ImportError:
            raise ImportError(
                "ConvNeXtV2+UPerNet requires transformers: pip install transformers"
            )
        
        self.num_classes = num_classes
        self.n_in = n_in
        
        # Map backbone names to HuggingFace model IDs
        arch = backbone_name.replace("convnextv2_", "")
        model_map = {
            "tiny": "openmmlab/upernet-convnext-tiny",
            "small": "openmmlab/upernet-convnext-small", 
            "base": "openmmlab/upernet-convnext-base",
            "large": "openmmlab/upernet-convnext-large",
        }
        model_name = model_map.get(arch, f"openmmlab/upernet-convnext-{arch}")
        
        if pretrained:
            self.model = AutoModelForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            if n_in != 3:
                self._adapt_input_channels(n_in)
        else:
            config = UperNetConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            self.model = AutoModelForSemanticSegmentation.from_config(config)
            if n_in != 3:
                self._adapt_input_channels(n_in)
    
    def _adapt_input_channels(self, n_in):
        """Adapt the model to handle different number of input channels"""
        try:
            # Update the config to reflect new number of channels
            if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'config'):
                self.model.backbone.config.num_channels = n_in
            
            if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'embeddings'):
                old_patch_embed = self.model.backbone.embeddings.patch_embeddings
                
                new_patch_embed = nn.Conv2d(
                    n_in,
                    old_patch_embed.out_channels,
                    kernel_size=old_patch_embed.kernel_size,
                    stride=old_patch_embed.stride,
                    padding=old_patch_embed.padding,
                    bias=old_patch_embed.bias is not None
                )
                
                nn.init.kaiming_normal_(new_patch_embed.weight, mode='fan_out', nonlinearity='relu')
                if new_patch_embed.bias is not None:
                    nn.init.constant_(new_patch_embed.bias, 0)
                
                if n_in >= 3 and old_patch_embed.weight.shape[1] == 3:
                    with torch.no_grad():
                        new_patch_embed.weight[:, :3] = old_patch_embed.weight
                
                self.model.backbone.embeddings.patch_embeddings = new_patch_embed
                
                # Also update num_channels in embeddings
                if hasattr(self.model.backbone.embeddings, 'num_channels'):
                    self.model.backbone.embeddings.num_channels = n_in
            else:
                print("Warning: Could not find patch embedding layer to adapt")
        except Exception as e:
            print(f"Warning: Could not adapt input channels: {e}")

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:],
                                mode='bilinear', align_corners=False)
        return logits


# ---------------------------------------------------------------------
# Training class
# ---------------------------------------------------------------------
class BasicTrainingFastai2:
    def __init__(self, cfg, dls):
        self.cfg = cfg
        self.learn = self._build_learner(cfg, dls)

    def _num_classes(self):
        """Get number of classes from config or codes file"""
        if "num_classes" in self.cfg:
            return self.cfg["num_classes"]
        if "n_classes" in self.cfg:
            return self.cfg["n_classes"]
        if "path_to_codes" in self.cfg:
            with open(self.cfg["path_to_codes"]) as f:
                class_names = [l.strip() for l in f if l.strip()]
                print(f"Loaded {len(class_names)} classes from codes file: {class_names}")
                return len(class_names)
        return None

    def _loss(self):
        """Create loss function based on config"""
        ignore = int(self.cfg.get("ignore_index", 255))
        lt = self.cfg.get("loss_function", "cross_entropy")
        
        weights = None
        if "class_weights" in self.cfg and self.cfg["class_weights"]:
            weights = torch.tensor(self.cfg["class_weights"]).cuda()
            print(f"Using class weights: {self.cfg['class_weights']}")

        if lt == "dice":
            return DiceLoss(ignore_index=ignore)
        if lt == "focal":
            return FocalLoss(ignore_index=ignore)
        if lt == "combined":
            return CombinedLoss(ignore_index=ignore, class_weights=weights)

        print(f"Using CrossEntropyLoss with ignore_index={ignore}")
        return CrossEntropyLossFlat(axis=1, ignore_index=ignore, weight=weights)

    def _metric(self, ignore):
        """Create accuracy metric that respects ignore_index"""
        def acc(inp, targ):
            targ = targ.squeeze(1)
            mask = targ != ignore
            return (inp.argmax(1)[mask] == targ[mask]).float().mean()
        return acc

    def _build_learner(self, cfg, dls):
        """Build the appropriate learner based on model type"""
        ignore = int(cfg.get("ignore_index", 255))
        loss_func = self._loss()
        metric = self._metric(ignore)
        model_id = cfg["model"]

        # ConvNeXt V2 + UPerNet
        if isinstance(model_id, str) and model_id.endswith("_upernet"):
            print("Building ConvNeXt V2 + UPerNet")
            model = ConvNeXtV2UPerNetWrapper(
                backbone_name=model_id.replace("_upernet", ""),
                num_classes=self._num_classes(),
                n_in=len(cfg["means"]),
                pretrained=cfg.get("pretrained", True),
            )
            learn = Learner(
                dls, model, loss_func=loss_func, metrics=metric,
                path=cfg["log_folder"], model_dir=cfg["model_folder"]
            )

        # Swin + UPerNet
        elif isinstance(model_id, str) and "swin" in model_id.lower() and "upernet" in model_id.lower():
            print("Building Swin + UPerNet")
            swin_models = {
                "swin-small-upernet": "openmmlab/upernet-swin-small",
                "swin-base-upernet": "openmmlab/upernet-swin-base",
                "swin-large-upernet": "openmmlab/upernet-swin-large",
            }
            model_name = swin_models.get(model_id.lower(), model_id)
            
            model = SwinUPerNetWrapper(
                model_name=model_name,
                num_classes=self._num_classes(),
                n_in=len(cfg["means"]),
                pretrained=cfg.get("pretrained", True),
                ignore_index=ignore
            )
            learn = Learner(
                dls, model, loss_func=loss_func, metrics=metric,
                path=cfg["log_folder"], model_dir=cfg["model_folder"]
            )

        # SegFormer
        elif isinstance(model_id, str) and model_id.startswith("segformer"):
            print("Building SegFormer")
            segformer_models = {
                "segformer-b0": "nvidia/segformer-b0-finetuned-ade-512-512",
                "segformer-b1": "nvidia/segformer-b1-finetuned-ade-512-512",
                "segformer-b2": "nvidia/segformer-b2-finetuned-ade-512-512",
                "segformer-b3": "nvidia/segformer-b3-finetuned-ade-512-512",
                "segformer-b4": "nvidia/segformer-b4-finetuned-ade-512-512",
                "segformer-b5": "nvidia/segformer-b5-finetuned-ade-640-640",
            }
            model_name = segformer_models.get(model_id, model_id)
            
            model = SegFormerWrapper(
                model_name=model_name,
                num_classes=self._num_classes(),
                n_in=len(cfg["means"]),
                pretrained=cfg.get("pretrained", True),
                ignore_index=ignore
            )
            learn = Learner(
                dls, model, loss_func=loss_func, metrics=metric,
                path=cfg["log_folder"], model_dir=cfg["model_folder"]
            )

        # timm UNet (EfficientNet, etc.)
        elif isinstance(model_id, str) and ("efficientnet" in model_id or "bottleneck" in cfg):
            print("Building timm UNet")
            learn = timm_unet_learner(
                dls, model_id,
                loss_func=loss_func,
                metrics=metric,
                n_in=len(cfg["means"]),
                bottleneck=cfg.get("bottleneck"),
                pretrained=cfg.get("pretrained", True),
                path=cfg["log_folder"],
                model_dir=cfg["model_folder"]
            )

        # fastai UNet (ResNet, etc.)
        else:
            print("Building fastai UNet")
            learn = unet_learner(
                dls, model_id,
                loss_func=loss_func,
                metrics=metric,
                n_in=len(cfg["means"]),
                path=cfg["log_folder"],
                model_dir=cfg["model_folder"]
            )

        return learn.to_fp16() if cfg.get("to_fp16", False) else learn

    def find_learning_rate(self, show_images=False):
        """Find optimal learning rate"""
        lr_min, lr_steep, lr_slide, lr_valley = self.learn.lr_find(
            suggest_funcs=(minimum, steep, slide, valley)
        )
        print(f"lr_min: {lr_min}")
        print(f"lr_steep: {lr_steep}")
        print(f"lr_slide: {lr_slide}")
        print(f"lr_valley: {lr_valley}")
        
        if show_images:
            print("Exit the graph to continue")
            import matplotlib.pyplot as plt
            plt.show()
        
        print(f"Using lr_valley as learning rate: {lr_valley}")
        return lr_valley

    def create_folders(self):
        """Create folders for models and logs"""
        pathlib.Path(self.cfg["model_folder"]).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.cfg["log_folder"]).mkdir(parents=True, exist_ok=True)

    def train(self, lr):
        """Train the model"""
        self.create_folders()
        
        # Load pretrained weights if specified
        if self.cfg.get("model_to_load"):
            print(f"Loading: {self.cfg['model_to_load']}")
            self.learn.load(str(self.cfg["model_to_load"]).rstrip(".pth"))
        
        # Freeze/unfreeze
        if self.cfg.get("freeze", False):
            self.learn.freeze()
        else:
            self.learn.unfreeze()
        
        # Setup callbacks
        n_batch = self.cfg.get("save_on_batch_iter_modulus_n", 0)
        start_epoch = self.cfg.get("last_epoch", -1) + 1
        
        cbs = [
            GradientAccumulation(self.cfg.get("n_acc", 1)),
            GradientClip(self.cfg.get("gradient_clip", 1.0)),
            SaveModelCallback(
                monitor='valid_loss',
                fname=self.cfg["job_name"],
                every_epoch=True,
                with_opt=True
            ),
            CSVLoggerWithLR(fname=self.cfg["job_name"] + ".csv", append=True),
        ]
        
        #if n_batch > 0:
        #    cbs.append(DoThingsAfterBatch(n_batch=n_batch))
        
        # Train with appropriate scheduler
        scheduler = self.cfg.get("scheduler", "fit_one_cycle")
        
        if scheduler == "fit_one_cycle":
            self.learn.fit_one_cycle(
                n_epoch=self.cfg["epochs"],
                start_epoch=start_epoch,
                lr_max=lr,
                cbs=cbs
            )
        elif scheduler == "fixed":
            self.learn.fit(
                n_epoch=self.cfg["epochs"],
                start_epoch=start_epoch,
                lr=lr,
                cbs=cbs
            )
        else:
            sys.exit(f"Unknown scheduler: {scheduler}")
        
        # Save final model
        print("Saving model")
        self.learn.save(self.cfg["job_name"])


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def train_experiment(cfg):
    """Run a training experiment"""
    dls = sdfi_dataset.get_dataset(cfg)
    trainer = BasicTrainingFastai2(cfg, dls)
    
    # Determine learning rate
    if "lr" in cfg:
        max_lr = cfg["lr"]
        print(f"Using predefined max learning rate: {max_lr}")
    else:
        print("Finding learning rate...")
        lr_valley = trainer.find_learning_rate(show_images=False)
        
        if cfg.get("scheduler", "fit_one_cycle") == "fit_one_cycle":
            multiply_with = 30
            print(f"Multiplying lr_valley by {multiply_with} for fit_one_cycle")
            max_lr = lr_valley * multiply_with
        else:
            max_lr = lr_valley
        
        cfg["lr_finder_lr"] = max_lr
    
    print(f"max_lr: {max_lr}")
    print(f"job_name: {cfg['job_name']}")
    
    trainer.train(max_lr)
    
    print(f"TRAINING DONE! job_name: {cfg['job_name']}")
    sdfi_utils.save_dictionary_to_disk(cfg)


def infer_model_and_log_folders(cfg):
    """Create model_folder and log_folder paths"""
    cfg['model_folder'] = (
        Path(cfg['experiment_root']) / 
        Path(cfg['job_name']) / 
        Path("models")
    ).resolve()
    cfg['log_folder'] = (
        Path(cfg['experiment_root']) / 
        Path(cfg['job_name']) / 
        Path("logs")
    ).resolve()


if __name__ == "__main__":
    usage_example = (
        "Example usage:\n"
        "python train.py --config configs/example_configs/train_example_dataset.ini\n"
        "To use another GPU: CUDA_VISIBLE_DEVICES=1 python train.py --config ...\n"
    )
    
    parser = argparse.ArgumentParser(
        epilog=usage_example,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-c", "--config", nargs="+", required=True,
                        help="One or more paths to experiment config files")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic training for reproducibility")
    
    args = parser.parse_args()

    for cfg_path in args.config:
        cfg = sdfi_utils.load_settings_from_config_file(cfg_path)
        
        if args.deterministic:
            cfg["num_workers"] = 1
            make_deterministic()
        
        infer_model_and_log_folders(cfg)
        train_experiment(cfg)
