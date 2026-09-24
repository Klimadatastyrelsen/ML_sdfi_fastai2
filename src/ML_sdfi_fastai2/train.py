#!/usr/bin/env python
# coding: utf-8
"""
Train a model to do semantic segmentation

Supported architectures:
- UNet (fastai)
- timm UNet (EfficientNet / ConvNeXt)
- SegFormer
- Swin + UPerNet
- ConvNeXt V1 + UPerNet
"""

import os
import sys
import time
import json
import math
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
from fastai.learner import Metric

import utils.utils as sdfi_utils
import sdfi_dataset

from wwf.vision.timm import timm_unet_learner

from utils.timm_unet_utils import timm_unet_splitter


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
    """fastai CSVLogger with one lr column per optimizer param group.

    fastai's CSVLogger writes each epoch row itself by hooking `learn.logger`
    (`_write_line`). Overriding `after_epoch` to write a second row, as an
    earlier version did, produced every epoch twice (once without and once
    with lr columns). We therefore extend `_write_line` and the header instead.
    """
    def _lrs(self):
        return [g['lr'] for g in self.learn.opt.param_groups]

    def before_fit(self):
        if hasattr(self, "gather_preds"):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = (self.path / self.fname).open('a' if self.append else 'w')
        lr_names = [f'lr_{i}' for i in range(len(self._lrs()))]
        self.file.write(','.join(list(self.recorder.metric_names) + lr_names) + '\n')
        self.old_logger, self.learn.logger = self.logger, self._write_line

    def _write_line(self, log):
        self.file.write(','.join(str(t) for t in list(log) + self._lrs()) + '\n')
        self.file.flush()
        os.fsync(self.file.fileno())
        self.old_logger(log)


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
# NaN guard: classify, react and report non-finite losses
# ---------------------------------------------------------------------
def _all_finite(t):
    return bool(torch.isfinite(t).all()) if isinstance(t, torch.Tensor) else True


class NaNGuard(Callback):
    """Detect non-finite training losses, name the cause, keep training and keep logs finite.

    Per training batch with a non-finite loss (checked in `after_loss`):
      input_nonfinite   nan/inf in the input batch        -> skip batch
      void_target       no pixel outside ignore_index     -> skip batch
      diverged          nan/inf in model weights          -> stop the fit (fatal)
      pred_nonfinite    nan/inf in the prediction         -> skip batch (fp16 overflow / spike)
      loss_numerics     everything else finite            -> skip batch (loss function itself)

    Skipping = `CancelBatchException` from `after_loss`: no backward, no step.
    `learn.loss` is replaced with the last finite loss so fastai's smoothed
    train_loss (an exponential moving average that is never reset) stays finite.

    Per epoch: prints a summary and appends a row to `<job_name>_nan_report.csv`
    in the log folder (learn.path).

    order=20 places this after MixedPrecision (10), whose `after_loss` must
    exit autocast before we can cancel the batch, and before Recorder (50).
    """
    order = 20
    CAUSES = ("input_nonfinite", "void_target", "diverged", "pred_nonfinite", "loss_numerics")

    # Causes for which the learning rate is a plausible driver.
    LR_CAUSES = ("diverged", "pred_nonfinite")
    # The lr at which the loss turns non-finite is a ceiling, not the culprit:
    # the weights drift over several earlier steps. Suggest well below it.
    LR_MARGIN = 0.5

    def __init__(self, ignore_index=255, report_name="nan_report.csv", max_prints_per_epoch=5,
                 lr_max=None, lr_source="config", lr_multiplier=None):
        self.ignore_index = int(ignore_index)
        self.report_name = report_name
        self.max_prints_per_epoch = max_prints_per_epoch
        self.lr_max = None if lr_max is None else float(lr_max)   # peak lr of the schedule
        self.lr_source = lr_source                                # "config" or "lr_finder"
        self.lr_multiplier = lr_multiplier                        # lr_valley multiplier when lr_finder

    def before_fit(self):
        self._last_finite_loss = None
        # Fit-level lr bookkeeping for the suggestion.
        self.max_lr_seen = 0.0
        self.min_nan_lr = None          # lowest lr at which an LR-related nan occurred during warm-up
        self.after_peak_events = 0      # LR-related nans that occurred after a higher lr was survived
        self.after_peak_survived_lr = None
        self._reset_epoch_counts()

    # ---- lr suggestion -------------------------------------------------
    @staticmethod
    def _round_sig(x, sig=2):
        return float(f"{x:.{sig}g}")

    def _record_lr_event(self, cause, lr):
        if cause not in self.LR_CAUSES or not math.isfinite(lr):
            return
        if lr < self.max_lr_seen * (1 - 1e-6):
            # Annealing phase: the model already survived a higher lr.
            self.after_peak_events += 1
            self.after_peak_survived_lr = self.max_lr_seen
        else:
            self.min_nan_lr = lr if self.min_nan_lr is None else min(self.min_nan_lr, lr)

    def suggested_lr(self):
        """Peak lr to stay below the first LR-related nan, or None if not applicable."""
        if self.min_nan_lr is None:
            return None
        return self._round_sig(self.min_nan_lr * self.LR_MARGIN)

    def lr_advice(self):
        """Human readable advice; '' when there is nothing LR-related to say."""
        parts = []
        s = self.suggested_lr()
        if s is not None:
            msg = (f"non-finite loss first appeared at lr={self.min_nan_lr:.3e} while the lr was rising"
                   f" (schedule peak {self.lr_max:.3e}, from {self.lr_source}). "
                   f"Suggestion (heuristic, margin {self.LR_MARGIN}): keep the peak lr at most {s:.3g}")
            if self.lr_source == "lr_finder" and self.lr_multiplier and self.lr_max:
                new_mult = self._round_sig(self.lr_multiplier * s / self.lr_max)
                msg += (f", i.e. lower the lr_valley multiplier in train_experiment from "
                        f"{self.lr_multiplier} to about {new_mult:g}, or set lr = {s:.3g} in the config")
            else:
                msg += f", i.e. set lr = {s:.3g} in the config"
            msg += (". Alternatives that keep the peak: longer warm-up (pct_start), stronger "
                    "gradient_clip, bf16 instead of fp16.")
            parts.append(msg)
        if self.after_peak_events:
            parts.append(f"{self.after_peak_events} LR-related nan(s) occurred after the model had already "
                         f"survived lr={self.after_peak_survived_lr:.3e}; the lr value is probably not the "
                         f"sole cause there (accumulated instability, fp16 range, or data).")
        return " ".join(parts)

    def _reset_epoch_counts(self):
        self.counts = {c: 0 for c in self.CAUSES}
        self.valid_nonfinite = 0
        self.first_iter = None
        self.first_lr = None
        self.first_cause = None
        self._n_printed = 0

    def _current_lr(self):
        try:
            return float(self.opt.hypers[-1]['lr'])
        except Exception:
            return float('nan')

    def _classify(self):
        if not all(_all_finite(x) for x in self.xb):
            return "input_nonfinite"
        targ = self.yb[0]
        if not bool((targ != self.ignore_index).any()):
            return "void_target"
        if not all(_all_finite(p) for p in self.learn.model.parameters()):
            return "diverged"
        if not _all_finite(self.pred):
            return "pred_nonfinite"
        return "loss_numerics"

    def after_loss(self):
        if len(self.yb) == 0:
            return
        loss = self.loss
        if not self.training:
            # Validation: never alter valid_loss (it is a plain per-epoch mean and
            # a nan there is real information), only count it for the report.
            if not _all_finite(loss):
                self.valid_nonfinite += 1
            return
        lr = self._current_lr()
        if math.isfinite(lr):
            self.max_lr_seen = max(self.max_lr_seen, lr)
        if _all_finite(loss):
            self._last_finite_loss = loss.detach().clone()
            return

        cause = self._classify()
        self.counts[cause] += 1
        self._record_lr_event(cause, lr)
        if self.first_iter is None:
            self.first_iter, self.first_lr, self.first_cause = int(self.iter), lr, cause
        if self._n_printed < self.max_prints_per_epoch:
            self._n_printed += 1
            print(f"[NaNGuard] epoch {self.epoch} iter {self.iter}/{self.n_iter} lr={lr:.3e}: "
                  f"non-finite loss, cause={cause}, batch skipped"
                  + (" -> STOPPING" if cause == "diverged" else ""))

        # Keep the logged/smoothed train loss finite.
        self.learn.loss = (self._last_finite_loss if self._last_finite_loss is not None
                           else torch.zeros_like(loss))

        if cause == "diverged":
            self._write_report_row(fatal=True)
            print("[NaNGuard] Model weights are non-finite. Training cannot recover; stopping fit.")
            advice = self.lr_advice()
            if advice:
                print("[NaNGuard] " + advice)
            raise CancelFitException()
        raise CancelBatchException()

    def _summary(self):
        return {c: self.counts[c] for c in self.CAUSES if self.counts[c]}

    def _write_report_row(self, fatal=False):
        path = Path(self.learn.path) / self.report_name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with open(path, 'a') as f:
            if write_header:
                f.write(','.join(['epoch', 'first_iter', 'first_lr', 'first_cause', 'fatal']
                                 + list(self.CAUSES) + ['valid_nonfinite',
                                 'config_lr', 'lr_source', 'suggested_lr']) + '\n')
            s = self.suggested_lr()
            f.write(','.join(map(str, [
                int(self.epoch),
                '' if self.first_iter is None else self.first_iter,
                '' if self.first_lr is None else f"{self.first_lr:.6e}",
                self.first_cause or '',
                int(fatal),
            ] + [self.counts[c] for c in self.CAUSES] + [
                self.valid_nonfinite,
                '' if self.lr_max is None else f"{self.lr_max:.6e}",
                self.lr_source,
                '' if s is None else f"{s:.6e}",
            ])) + '\n')

    def after_epoch(self):
        total = sum(self.counts.values())
        if total:
            print(f"[NaNGuard] epoch {self.epoch}: {total} non-finite training batch(es) skipped, "
                  f"first at iter {self.first_iter} (lr={self.first_lr:.3e}), causes={self._summary()}")
            if any(self.counts[c] for c in self.LR_CAUSES):
                advice = self.lr_advice()
                if advice:
                    print("[NaNGuard] " + advice)
        if self.valid_nonfinite:
            print(f"[NaNGuard] epoch {self.epoch}: {self.valid_nonfinite} validation batch(es) had a "
                  f"non-finite loss; valid_loss for this epoch is not reliable (check validation data).")
        if total or self.valid_nonfinite:
            self._write_report_row(fatal=False)
        self._reset_epoch_counts()

    def after_fit(self):
        advice = self.lr_advice()
        if advice:
            print("[NaNGuard] fit summary: " + advice)


class SkipNonFiniteGradStep(Callback):
    """fp32 only: skip the optimizer step when any gradient is non-finite.

    Under fp16, MixedPrecision/GradScaler already skips such steps. Under fp32
    nothing does, and GradientClip would multiply every gradient by a nan norm.
    order=9 runs before MixedPrecision (10) and GradientClip (11); we do nothing
    when a GradScaler is active because cancelling before `scaler.step` breaks
    `scaler.update`.
    """
    order = 9

    def before_fit(self):
        self.n_skipped = 0

    def before_step(self):
        if getattr(self.learn, 'scaler', None) is not None:
            return
        for p in self.learn.model.parameters():
            if p.grad is not None and not _all_finite(p.grad):
                self.n_skipped += 1
                print(f"[NaNGuard] epoch {self.epoch} iter {self.iter}: finite loss but non-finite "
                      f"gradient, optimizer step skipped (total {self.n_skipped})")
                self.learn.opt.zero_grad()
                raise CancelStepException()


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


def _zero_if_all_ignored(loss, pred, target, ignore_index):
    """Return a finite 0 loss when no pixel in the batch counts towards the loss.

    nn.CrossEntropyLoss(reduction='mean') returns nan for a batch where every
    target pixel equals ignore_index. Its gradient is already zero, so training
    is unaffected, but fastai's smoothed train_loss is an exponential moving
    average that is never reset during fit. One nan poisons it for the rest
    of the run. Replacing it with 0 keeps the logged train_loss meaningful.

    The zero is built from `pred` (not from the nan loss) so it is finite and
    still attached to the graph, letting `backward()` run with zero gradients.
    """
    if bool((target != ignore_index).any()):
        return loss
    return pred.sum() * 0.0


class CrossEntropyLossFlatSafe(CrossEntropyLossFlat):
    """CrossEntropyLossFlat that returns 0 instead of nan for all-ignored batches."""

    def __call__(self, inp, targ, **kwargs):
        loss = super().__call__(inp, targ, **kwargs)
        return _zero_if_all_ignored(loss, inp, targ, self.func.ignore_index)


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5, ignore_index=255, class_weights=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        target_long = target.squeeze(1).long()
        ce = _zero_if_all_ignored(self.ce_loss(pred, target_long), pred, target_long, self.ignore_index)
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
# ConvNeXt V1 + UPerNet (transformers-based)
# ---------------------------------------------------------------------
class ConvNeXt1UPerNetWrapper(nn.Module):
    """ConvNeXt V1 backbone + UPerNet decoder using transformers library"""
    def __init__(self, backbone_name, num_classes, n_in, pretrained=True):
        super().__init__()
        try:
            from transformers import AutoModelForSemanticSegmentation, UperNetConfig
        except ImportError:
            raise ImportError(
                "ConvNeXt V1 + UPerNet requires transformers: pip install transformers"
            )
        
        self.num_classes = num_classes
        self.n_in = n_in
        
        # Map backbone names to HuggingFace model IDs
        arch = backbone_name.replace("convnext1_", "")
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
class PixelAccuracy(Metric):
    """Global pixel accuracy over non-ignored labels per validation epoch."""

    def __init__(self, ignore_index=0):
        self.ignore_index = int(ignore_index)

    def reset(self):
        self.correct, self.total = 0, 0

    def accumulate(self, learn):
        inp, targ = learn.pred, learn.y
        targ = targ.squeeze(1)
        mask = targ != self.ignore_index
        n = int(mask.sum())
        if n == 0:
            return
        self.correct += int((inp.argmax(1)[mask] == targ[mask]).sum())
        self.total += n

    @property
    def value(self):
        return self.correct / self.total if self.total else 0.0


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
        return CrossEntropyLossFlatSafe(axis=1, ignore_index=ignore, weight=weights)

    def _metric(self, ignore):
        """Create accuracy metric that respects ignore_index"""
        return PixelAccuracy(ignore_index=ignore)

    def _build_learner(self, cfg, dls):
        """Build the appropriate learner based on model type"""
        ignore = int(cfg.get("ignore_index", 255))
        loss_func = self._loss()
        metric = self._metric(ignore)
        model_id = cfg["model"]

        # ConvNeXt V1 + UPerNet
        if isinstance(model_id, str) and model_id.startswith("convnext1_") and model_id.endswith("_upernet"):
            print("Building ConvNeXt V1 + UPerNet")
            model = ConvNeXt1UPerNetWrapper(
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
                splitter=timm_unet_splitter,
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
            self.learn.load(str(pathlib.Path(self.cfg["model_to_load"]).with_suffix("")))
        
        # Freeze/unfreeze
        if self.cfg.get("freeze", False):
            self.learn.freeze()
        else:
            self.learn.unfreeze()
        
        # Setup callbacks
        n_batch = self.cfg.get("save_on_batch_iter_modulus_n", 0)
        # last_epoch = false / missing means a fresh run. (`False + 1 == 1` would
        # otherwise make SkipToEpoch skip epoch 0 of the schedule.)
        last_epoch = self.cfg.get("last_epoch", -1)
        if last_epoch is False or last_epoch is None:
            last_epoch = -1
        start_epoch = int(last_epoch) + 1
        
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
            NaNGuard(
                ignore_index=int(self.cfg.get("ignore_index", 255)),
                report_name=self.cfg["job_name"] + "_nan_report.csv",
                lr_max=lr,
                lr_source="config" if "lr" in self.cfg else "lr_finder",
                lr_multiplier=self.cfg.get("lr_finder_multiplier"),
            ),
            SkipNonFiniteGradStep(),
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
            multiply_with = 1
            max_lr = lr_valley
        
        cfg["lr_finder_lr"] = max_lr
        cfg["lr_finder_multiplier"] = multiply_with
    
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
