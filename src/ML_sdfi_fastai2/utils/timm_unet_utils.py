"""Patches for wwf timm UNet used with current fastcore/timm versions."""

from __future__ import annotations

import torch.nn as nn
from fastai.basics import params
from fastai.torch_core import getattrs
from fastcore.foundation import L
from wwf.vision.timm import split_nested_list


def _get_params_from_attrs(m, ls):
    return params(nn.Sequential(*getattrs(m, *ls)))


def _get_params_from_modules(modules):
    return params(nn.Sequential(*modules))


def timm_unet_splitter(m):
    """
    Parameter splitter for wwf TimmUnet.

    wwf's _timm_splitter uses L(m.encoder._modules), which with modern fastcore
    wraps the dict as a single item instead of listing module names. That breaks
    freeze()/create_opt for timm backbones (notably EfficientNetV2).
    """
    encoder_module_names = L(*m.encoder._modules.keys())
    encoder_split_idxs = m.feature_info.module_name(0).split(".")
    encoder_split_idxs[0] = encoder_module_names.index(encoder_split_idxs[0])
    encoder_modules = getattrs(m.encoder, *encoder_module_names)
    encoder_early, encoder_late = split_nested_list(encoder_modules, encoder_split_idxs)
    encoder_early = _get_params_from_modules(encoder_early)
    encoder_late = _get_params_from_modules(encoder_late)
    decoder = _get_params_from_attrs(m.decoder, L(*m.decoder._modules.keys()))
    head = _get_params_from_attrs(m.head, L(*m.head._modules.keys()))
    return L(encoder_early, encoder_late, L(*decoder, *head))
