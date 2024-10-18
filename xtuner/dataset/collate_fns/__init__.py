# Copyright (c) OpenMMLab. All rights reserved.
from .default_collate_fn import default_collate_fn
from .mmlu_collate_fn import mmlu_collate_fn
from .s2s_transformer_fn import s2s_transformer_fn

__all__ = ['default_collate_fn', 'mmlu_collate_fn', 's2s_transformer_fn']
