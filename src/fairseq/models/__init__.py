# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import argparse
import importlib
import os

from contextlib import ExitStack

from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import merge_with_parent
from omegaconf import open_dict, OmegaConf

from .composite_encoder import CompositeEncoder
from .distributed_fairseq_model import DistributedFairseqModel
from .fairseq_decoder import FairseqDecoder
from .fairseq_encoder import FairseqEncoder
from .fairseq_incremental_decoder import FairseqIncrementalDecoder
from .fairseq_model import (
    BaseFairseqModel,
    FairseqEncoderDecoderModel,
    FairseqEncoderModel,
    FairseqLanguageModel,
    FairseqModel,
    FairseqMultiModel,
)


MODEL_REGISTRY = {}
MODEL_DATACLASS_REGISTRY = {}
ARCH_MODEL_REGISTRY = {}
ARCH_MODEL_NAME_REGISTRY = {}
ARCH_MODEL_INV_REGISTRY = {}
ARCH_CONFIG_REGISTRY = {}


__all__ = [
    "BaseFairseqModel",
    "CompositeEncoder",
    "DistributedFairseqModel",
    "FairseqDecoder",
    "FairseqEncoder",
    "FairseqEncoderDecoderModel",
    "FairseqEncoderModel",
    "FairseqIncrementalDecoder",
    "FairseqLanguageModel",
    "FairseqModel",
    "FairseqMultiModel",
]
