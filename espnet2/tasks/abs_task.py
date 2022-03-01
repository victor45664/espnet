"""Abstract task module."""
from abc import ABC
from abc import abstractmethod
import argparse
from dataclasses import dataclass
from distutils.version import LooseVersion
import functools
import logging
import os
from pathlib import Path
import sys
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import humanfriendly
import numpy as np
import torch
import torch.multiprocessing
import torch.nn
import torch.optim
from torch.utils.data import DataLoader
from typeguard import check_argument_types
from typeguard import check_return_type
import yaml

from espnet import __version__
from espnet.utils.cli_utils import get_commandline_args
from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.iterators.chunk_iter_factory import ChunkIterFactory
from espnet2.iterators.multiple_iter_factory import MultipleIterFactory
from espnet2.iterators.sequence_iter_factory import SequenceIterFactory
from espnet2.main_funcs.collect_stats import collect_stats
from espnet2.optimizers.sgd import SGD
from espnet2.samplers.build_batch_sampler import BATCH_TYPES
from espnet2.samplers.build_batch_sampler import build_batch_sampler
from espnet2.samplers.unsorted_batch_sampler import UnsortedBatchSampler
from espnet2.schedulers.noam_lr import NoamLR
from espnet2.schedulers.warmup_lr import WarmupLR
from espnet2.torch_utils.load_pretrained_model import load_pretrained_model
from espnet2.torch_utils.model_summary import model_summary
from espnet2.torch_utils.pytorch_version import pytorch_cudnn_version
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.dataset import AbsDataset
from espnet2.train.dataset import DATA_TYPES
from espnet2.train.dataset import ESPnetDataset
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.distributed_utils import free_port
from espnet2.train.distributed_utils import get_master_port
from espnet2.train.distributed_utils import get_node_rank
from espnet2.train.distributed_utils import get_num_nodes
from espnet2.train.distributed_utils import resolve_distributed_mode
from espnet2.train.iterable_dataset import IterableESPnetDataset
from espnet2.train.trainer import Trainer
from espnet2.utils.build_dataclass import build_dataclass
from espnet2.utils import config_argparse
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import humanfriendly_parse_size_or_none
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_int
from espnet2.utils.types import str_or_none
from espnet2.utils.yaml_no_alias_safe_dump import yaml_no_alias_safe_dump

try:
    import wandb
except Exception:
    wandb = None

if LooseVersion(torch.__version__) >= LooseVersion("1.5.0"):
    from torch.multiprocessing.spawn import ProcessContext
else:
    from torch.multiprocessing.spawn import SpawnContext as ProcessContext


optim_classes = dict(
    adam=torch.optim.Adam,
    adamw=torch.optim.AdamW,
    sgd=SGD,
    adadelta=torch.optim.Adadelta,
    adagrad=torch.optim.Adagrad,
    adamax=torch.optim.Adamax,
    asgd=torch.optim.ASGD,
    lbfgs=torch.optim.LBFGS,
    rmsprop=torch.optim.RMSprop,
    rprop=torch.optim.Rprop,
)
if LooseVersion(torch.__version__) >= LooseVersion("1.10.0"):
    # From 1.10.0, RAdam is officially supported
    optim_classes.update(
        radam=torch.optim.RAdam,
    )
try:
    import torch_optimizer

    optim_classes.update(
        accagd=torch_optimizer.AccSGD,
        adabound=torch_optimizer.AdaBound,
        adamod=torch_optimizer.AdaMod,
        diffgrad=torch_optimizer.DiffGrad,
        lamb=torch_optimizer.Lamb,
        novograd=torch_optimizer.NovoGrad,
        pid=torch_optimizer.PID,
        # torch_optimizer<=0.0.1a10 doesn't support
        # qhadam=torch_optimizer.QHAdam,
        qhm=torch_optimizer.QHM,
        sgdw=torch_optimizer.SGDW,
        yogi=torch_optimizer.Yogi,
    )
    if LooseVersion(torch_optimizer.__version__) < LooseVersion("0.2.0"):
        # From 0.2.0, RAdam is dropped
        optim_classes.update(
            radam=torch_optimizer.RAdam,
        )
    del torch_optimizer
except ImportError:
    pass
try:
    import apex

    optim_classes.update(
        fusedadam=apex.optimizers.FusedAdam,
        fusedlamb=apex.optimizers.FusedLAMB,
        fusednovograd=apex.optimizers.FusedNovoGrad,
        fusedsgd=apex.optimizers.FusedSGD,
    )
    del apex
except ImportError:
    pass
try:
    import fairscale
except ImportError:
    fairscale = None


scheduler_classes = dict(
    ReduceLROnPlateau=torch.optim.lr_scheduler.ReduceLROnPlateau,
    lambdalr=torch.optim.lr_scheduler.LambdaLR,
    steplr=torch.optim.lr_scheduler.StepLR,
    multisteplr=torch.optim.lr_scheduler.MultiStepLR,
    exponentiallr=torch.optim.lr_scheduler.ExponentialLR,
    CosineAnnealingLR=torch.optim.lr_scheduler.CosineAnnealingLR,
    noamlr=NoamLR,
    warmuplr=WarmupLR,
    cycliclr=torch.optim.lr_scheduler.CyclicLR,
    onecyclelr=torch.optim.lr_scheduler.OneCycleLR,
    CosineAnnealingWarmRestarts=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
)
# To lower keys
optim_classes = {k.lower(): v for k, v in optim_classes.items()}
scheduler_classes = {k.lower(): v for k, v in scheduler_classes.items()}


@dataclass
class IteratorOptions:
    preprocess_fn: callable
    collate_fn: callable
    data_path_and_name_and_type: list
    shape_files: list
    batch_size: int
    batch_bins: int
    batch_type: str
    max_cache_size: float
    max_cache_fd: int
    distributed: bool
    num_batches: Optional[int]
    num_iters_per_epoch: Optional[int]
    train: bool


class AbsTask(ABC):
    # Use @staticmethod, or @classmethod,
    # instead of instance method to avoid God classes

    # If you need more than one optimizers, change this value in inheritance
    num_optimizers: int = 1
    trainer = Trainer
    class_choices_list: List[ClassChoices] = []

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    @abstractmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        pass

    @classmethod
    @abstractmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[[Sequence[Dict[str, np.ndarray]]], Dict[str, torch.Tensor]]:
        """Return "collate_fn", which is a callable object and given to DataLoader.

        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(collate_fn=cls.build_collate_fn(args, train=True), ...)

        In many cases, you can use our common collate_fn.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """Define the required names by Task

        This function is used by
        >>> cls.check_task_requirements()
        If your model is defined as following,

        >>> from espnet2.train.abs_espnet_model import AbsESPnetModel
        >>> class Model(AbsESPnetModel):
        ...     def forward(self, input, output, opt=None):  pass

        then "required_data_names" should be as

        >>> required_data_names = ('input', 'output')
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """Define the optional names by Task

        This function is used by
        >>> cls.check_task_requirements()
        If your model is defined as follows,

        >>> from espnet2.train.abs_espnet_model import AbsESPnetModel
        >>> class Model(AbsESPnetModel):
        ...     def forward(self, input, output, opt=None):  pass

        then "optional_data_names" should be as

        >>> optional_data_names = ('opt',)
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_model(cls, args: argparse.Namespace) -> AbsESPnetModel:
        raise NotImplementedError

    @classmethod
    def get_parser(cls) -> config_argparse.ArgumentParser:
        assert check_argument_types()

        class ArgumentDefaultsRawTextHelpFormatter(
            argparse.RawTextHelpFormatter,
            argparse.ArgumentDefaultsHelpFormatter,
        ):
            pass

        parser = config_argparse.ArgumentParser(
            description="base parser",
            formatter_class=ArgumentDefaultsRawTextHelpFormatter,
        )

        # NOTE(kamo): Use '_' instead of '-' to avoid confusion.
        #  I think '-' looks really confusing if it's written in yaml.

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        #  to provide --print_config mode. Instead of it, do as
        parser.set_defaults(required=["output_dir"])

        group = parser.add_argument_group("Common configuration")

        group.add_argument(
            "--print_config",
            action="store_true",
            help="Print the config file and exit",
        )
        group.add_argument(
            "--log_level",
            type=lambda x: x.upper(),
            default="INFO",
            choices=("ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
            help="The verbose level of logging",
        )
        group.add_argument(
            "--dry_run",
            type=str2bool,
            default=False,
            help="Perform process without training",
        )
        group.add_argument(
            "--iterator_type",
            type=str,
            choices=["sequence", "chunk", "task", "none"],
            default="sequence",
            help="Specify iterator type",
        )

        group.add_argument("--output_dir", type=str_or_none, default=None)
        group.add_argument(
            "--ngpu",
            type=int,
            default=0,
            help="The number of gpus. 0 indicates CPU mode",
        )
        group.add_argument("--seed", type=int, default=0, help="Random seed")
        group.add_argument(
            "--num_workers",
            type=int,
            default=1,
            help="The number of workers used for DataLoader",
        )
        group.add_argument(
            "--num_att_plot",
            type=int,
            default=3,
            help="The number images to plot the outputs from attention. "
            "This option makes sense only when attention-based model. "
            "We can also disable the attention plot by setting it 0",
        )

        group = parser.add_argument_group("distributed training related")
        group.add_argument(
            "--dist_backend",
            default="nccl",
            type=str,
            help="distributed backend",
        )
        group.add_argument(
            "--dist_init_method",
            type=str,
            default="env://",
            help='if init_method="env://", env values of "MASTER_PORT", "MASTER_ADDR", '
            '"WORLD_SIZE", and "RANK" are referred.',
        )
        group.add_argument(
            "--dist_world_size",
            default=None,
            type=int_or_none,
            help="number of nodes for distributed training",
        )
        group.add_argument(
            "--dist_rank",
            type=int_or_none,
            default=None,
            help="node rank for distributed training",
        )
        group.add_argument(
            # Not starting with "dist_" for compatibility to launch.py
            "--local_rank",
            type=int_or_none,
            default=None,
            help="local rank for distributed training. This option is used if "
            "--multiprocessing_distributed=false",
        )
        group.add_argument(
            "--dist_master_addr",
            default=None,
            type=str_or_none,
            help="The master address for distributed training. "
            "This value is used when dist_init_method == 'env://'",
        )
        group.add_argument(
            "--dist_master_port",
            default=None,
            type=int_or_none,
            help="The master port for distributed training"
            "This value is used when dist_init_method == 'env://'",
        )
        group.add_argument(
            "--dist_launcher",
            default=None,
            type=str_or_none,
            choices=["slurm", "mpi", None],
            help="The launcher type for distributed training",
        )
        group.add_argument(
            "--multiprocessing_distributed",
            default=False,
            type=str2bool,
            help="Use multi-processing distributed training to launch "
            "N processes per node, which has N GPUs. This is the "
            "fastest way to use PyTorch for either single node or "
            "multi node data parallel training",
        )
        group.add_argument(
            "--unused_parameters",
            type=str2bool,
            default=False,
            help="Whether to use the find_unused_parameters in "
            "torch.nn.parallel.DistributedDataParallel ",
        )
        group.add_argument(
            "--sharded_ddp",
            default=False,
            type=str2bool,
            help="Enable sharded training provided by fairscale",
        )

        group = parser.add_argument_group("cudnn mode related")
        group.add_argument(
            "--cudnn_enabled",
            type=str2bool,
            default=torch.backends.cudnn.enabled,
            help="Enable CUDNN",
        )
        group.add_argument(
            "--cudnn_benchmark",
            type=str2bool,
            default=torch.backends.cudnn.benchmark,
            help="Enable cudnn-benchmark mode",
        )
        group.add_argument(
            "--cudnn_deterministic",
            type=str2bool,
            default=True,
            help="Enable cudnn-deterministic mode",
        )

        group = parser.add_argument_group("collect stats mode related")
        group.add_argument(
            "--collect_stats",
            type=str2bool,
            default=False,
            help='Perform on "collect stats" mode',
        )
        group.add_argument(
            "--write_collected_feats",
            type=str2bool,
            default=False,
            help='Write the output features from the model when "collect stats" mode',
        )

        group = parser.add_argument_group("Trainer related")
        group.add_argument(
            "--max_epoch",
            type=int,
            default=40,
            help="The maximum number epoch to train",
        )
        group.add_argument(
            "--patience",
            type=int_or_none,
            default=None,
            help="Number of epochs to wait without improvement "
            "before stopping the training",
        )
        group.add_argument(
            "--val_scheduler_criterion",
            type=str,
            nargs=2,
            default=("valid", "loss"),
            help="The criterion used for the value given to the lr scheduler. "
            'Give a pair referring the phase, "train" or "valid",'
            'and the criterion name. The mode specifying "min" or "max" can '
            "be changed by --scheduler_conf",
        )
        group.add_argument(
            "--early_stopping_criterion",
            type=str,
            nargs=3,
            default=("valid", "loss", "min"),
            help="The criterion used for judging of early stopping. "
            'Give a pair referring the phase, "train" or "valid",'
            'the criterion name and the mode, "min" or "max", e.g. "acc,max".',
        )
        group.add_argument(
            "--best_model_criterion",
        