import argparse
import logging
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type
from espnet2.tasks.abs_task import IteratorOptions
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,  # noqa: H301
)
from espnet2.asr.decoder.transformer_decoder import DynamicConvolutionTransformerDecoder
from espnet2.asr.decoder.transformer_decoder import (
    LightweightConvolution2DTransformerDecoder,  # noqa: H301
)
from espnet2.asr.decoder.transformer_decoder import (
    LightweightConvolutionTransformerDecoder,  # noqa: H301
)
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.hubert_encoder import FairseqHubertEncoder
from espnet2.asr.encoder.hubert_encoder import FairseqHubertPretrainEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.contextual_block_transformer_encoder import (
    ContextualBlockTransformerEncoder,  # noqa: H301
)
from espnet2.asr.encoder.contextual_block_conformer_encoder import (
    ContextualBlockConformerEncoder,  # noqa: H301
)
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.encoder.wav2vec2_encoder import FairSeqWav2Vec2Encoder
from espnet2.asr.espnet_model import ESPnetASRModel,ESPnetASRModel_kd,ESPnetASRModel_unadl
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.postencoder.hugging_face_transformers_postencoder import (
    HuggingFaceTransformersPostEncoder,  # noqa: H301
)
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.asr.preencoder.sinc import LightweightSincConvs
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.utils.build_dataclass import build_dataclass
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import float_or_none
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none
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
from espnet2.train.trainer2 import Trainer_ilmt
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
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.abs_task import scheduler_classes
import wandb
frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
    ),
    type_check=AbsFrontend,
    default="default",
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(specaug=SpecAug),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default="utterance_mvn",
    optional=True,
)
preencoder_choices = ClassChoices(
    name="preencoder",
    classes=dict(
        sinc=LightweightSincConvs,
        linear=LinearProjection,
    ),
    type_check=AbsPreEncoder,
    default=None,
    optional=True,
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        contextual_block_transformer=ContextualBlockTransformerEncoder,
        contextual_block_conformer=ContextualBlockConformerEncoder,
        vgg_rnn=VGGRNNEncoder,
        rnn=RNNEncoder,
        wav2vec2=FairSeqWav2Vec2Encoder,
        hubert=FairseqHubertEncoder,
        hubert_pretrain=FairseqHubertPretrainEncoder,
    ),
    type_check=AbsEncoder,
    default="rnn",
)
postencoder_choices = ClassChoices(
    name="postencoder",
    classes=dict(
        hugging_face_transformers=HuggingFaceTransformersPostEncoder,
    ),
    type_check=AbsPostEncoder,
    default=None,
    optional=True,
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        transformer=TransformerDecoder,
        lightweight_conv=LightweightConvolutionTransformerDecoder,
        lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
        dynamic_conv=DynamicConvolutionTransformerDecoder,
        dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
        rnn=RNNDecoder,
    ),
    type_check=AbsDecoder,
    default="rnn",
)



class ASRTask_ilmt(ASRTask):


    @classmethod
    def check_task_requirements(
        cls,
        dataset: Union[AbsDataset, IterableESPnetDataset],
        allow_variable_data_keys: bool,
        train: bool,
        inference: bool = False,
    ) -> None:
        pass  #不要check require，这是一个work around
    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        super().add_task_arguments(parser)
        group = parser.add_argument_group(description="ilme related")
        group.add_argument(
            f"--ilme_conf",
            action=NestedDictAction,
            default=dict(),
            help=f"The keyword arguments for ilme",
        )

        group.add_argument("--ilmt_lm_data", type=str, action="append", default=[])
        group.add_argument("--ilmt_lm_data_shape_file", type=str, action="append", default=[])
        group.add_argument("--ilmt_loss_weight", type=float,  default=1,help="Ilmt Loss Weight this is")

        group.add_argument(
            "--ilmt_batch_bins",
            type=int,
            default=1000000,
            help="The number of batch bins. Used if batch_type='length' or 'numel'",
        )

        group.add_argument(
            "--ilmt_accum_grad",
            type=int,
            default=1,
            help="The number of gradient accumulation of ILMT",
        )
    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetASRModel:
         model=super().build_model(args)
         if "ilme_conf" in args:
            model.decoder.init_ilme(args.ilme_conf)
         else:
             logging.warning("ilme config is not present decoding as normal model")
         return model


    @classmethod
    def main_worker(cls, args: argparse.Namespace):
        assert check_argument_types()

        # 0. Init distributed process
        distributed_option = build_dataclass(DistributedOption, args)
        # Setting distributed_option.dist_rank, etc.
        distributed_option.init_options()

        # NOTE(kamo): Don't use logging before invoking logging.basicConfig()
        if not distributed_option.distributed or distributed_option.dist_rank == 0:
            if not distributed_option.distributed:
                _rank = ""
            else:
                _rank = (
                    f":{distributed_option.dist_rank}/"
                    f"{distributed_option.dist_world_size}"
                )

            # NOTE(kamo):
            # logging.basicConfig() is invoked in main_worker() instead of main()
            # because it can be invoked only once in a process.
            # FIXME(kamo): Should we use logging.getLogger()?
            logging.basicConfig(
                level=args.log_level,
                format=f"[{os.uname()[1].split('.')[0]}{_rank}]"
                f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
            )
        else:
            # Suppress logging if RANK != 0
            logging.basicConfig(
                level="ERROR",
                format=f"[{os.uname()[1].split('.')[0]}"
                f":{distributed_option.dist_rank}/{distributed_option.dist_world_size}]"
                f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
            )
        # Invoking torch.distributed.init_process_group
        distributed_option.init_torch_distributed()

        # 1. Set random-seed
        set_all_random_seed(args.seed)
        torch.backends.cudnn.enabled = args.cudnn_enabled
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        torch.backends.cudnn.deterministic = args.cudnn_deterministic
        if args.detect_anomaly:
            logging.info("Invoking torch.autograd.set_detect_anomaly(True)")
            torch.autograd.set_detect_anomaly(args.detect_anomaly)

        # 2. Build model
        model = cls.build_model(args=args)
        if not isinstance(model, AbsESPnetModel):
            raise RuntimeError(
                f"model must inherit {AbsESPnetModel.__name__}, but got {type(model)}"
            )
        model = model.to(
            dtype=getattr(torch, args.train_dtype),
            device="cuda" if args.ngpu > 0 else "cpu",
        )
        for t in args.freeze_param:
            for k, p in model.named_parameters():
                if k.startswith(t + ".") or k == t:
                    logging.info(f"Setting {k}.requires_grad = False")
                    p.requires_grad = False

        # 3. Build optimizer
        optimizers = cls.build_optimizers(args, model=model)

        # 4. Build schedulers
        schedulers = []
        for i, optim in enumerate(optimizers, 1):
            suf = "" if i == 1 else str(i)
            name = getattr(args, f"scheduler{suf}")
            conf = getattr(args, f"scheduler{suf}_conf")
            if name is not None:
                cls_ = scheduler_classes.get(name)
                if cls_ is None:
                    raise ValueError(
                        f"must be one of {list(scheduler_classes)}: {name}"
                    )
                scheduler = cls_(optim, **conf)
            else:
                scheduler = None

            schedulers.append(scheduler)

        logging.info(pytorch_cudnn_version())
        logging.info(model_summary(model))
        for i, (o, s) in enumerate(zip(optimizers, schedulers), 1):
            suf = "" if i == 1 else str(i)
            logging.info(f"Optimizer{suf}:\n{o}")
            logging.info(f"Scheduler{suf}: {s}")

        # 5. Dump "args" to config.yaml
        # NOTE(kamo): "args" should be saved after object-buildings are done
        #  because they are allowed to modify "args".
        output_dir = Path(args.output_dir)
        if not distributed_option.distributed or distributed_option.dist_rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
                logging.info(
                    f'Saving the configuration in {output_dir / "config.yaml"}'
                )
                yaml_no_alias_safe_dump(vars(args), f, indent=4, sort_keys=False)

        if args.dry_run:
            pass
        elif args.collect_stats:
            # Perform on collect_stats mode. This mode has two roles
            # - Derive the length and dimension of all input data
            # - Accumulate feats, square values, and the length for whitening
            logging.info(args)

            if args.valid_batch_size is None:
                args.valid_batch_size = args.batch_size

            if len(args.train_shape_file) != 0:
                train_key_file = args.train_shape_file[0]
            else:
                train_key_file = None
            if len(args.valid_shape_file) != 0:
                valid_key_file = args.valid_shape_file[0]
            else:
                valid_key_file = None

            collect_stats(
                model=model,
                train_iter=cls.build_streaming_iterator(
                    data_path_and_name_and_type=args.train_data_path_and_name_and_type,
                    key_file=train_key_file,
                    batch_size=args.batch_size,
                    dtype=args.train_dtype,
                    num_workers=args.num_workers,
                    allow_variable_data_keys=args.allow_variable_data_keys,
                    ngpu=args.ngpu,
                    preprocess_fn=cls.build_preprocess_fn(args, train=False),
                    collate_fn=cls.build_collate_fn(args, train=False),
                ),
                valid_iter=cls.build_streaming_iterator(
                    data_path_and_name_and_type=args.valid_data_path_and_name_and_type,
                    key_file=valid_key_file,
                    batch_size=args.valid_batch_size,
                    dtype=args.train_dtype,
                    num_workers=args.num_workers,
                    allow_variable_data_keys=args.allow_variable_data_keys,
                    ngpu=args.ngpu,
                    preprocess_fn=cls.build_preprocess_fn(args, train=False),
                    collate_fn=cls.build_collate_fn(args, train=False),
                ),
                output_dir=output_dir,
                ngpu=args.ngpu,
                log_interval=args.log_interval,
                write_collected_feats=args.write_collected_feats,
            )
        else:
            # 6. Loads pre-trained model
            for p in args.init_param:
                logging.info(f"Loading pretrained params from {p}")
                load_pretrained_model(
                    model=model,
                    init_param=p,
                    ignore_init_mismatch=args.ignore_init_mismatch,
                    # NOTE(kamo): "cuda" for torch.load always indicates cuda:0
                    #   in PyTorch<=1.4
                    map_location=f"cuda:{torch.cuda.current_device()}"
                    if args.ngpu > 0
                    else "cpu",
                )

            # 7. Build iterator factories
            if args.multiple_iterator:
                train_iter_factory = cls.build_multiple_iter_factory(
                    args=args,
                    distributed_option=distributed_option,
                    mode="train",
                )
            else:
                train_iter_factory = cls.build_iter_factory(
                    args=args,
                    distributed_option=distributed_option,
                    mode="train",
                )

            valid_iter_factory = cls.build_iter_factory(
                args=args,
                distributed_option=distributed_option,
                mode="valid",
            )

            train_num_of_iter=len(train_iter_factory.sampler.batches) \
                if train_iter_factory.num_iters_per_epoch==None \
                else train_iter_factory.num_iters_per_epoch     #train 一个epoch的迭代次数
            train_num_of_iter=train_num_of_iter*args.ilmt_accum_grad   #需要乘以ilmt的梯度叠加倍数，需要注意如果这里太大可能会出问题，不过一般应该遇不到？


            ilmt_iter_options=cls.build_ilmt_iter_options(args,distributed_option,"train",train_num_of_iter)
            ilmt_train_iter_factory =cls.build_sequence_iter_factory(
                args=args,
                iter_options=ilmt_iter_options,
                mode="train",
            )   #ilmt dataloader，这个用来读取外部文本数据，与主loader不同步

            #vdebug:
            # temp = ilmt_train_iter_factory.build_iter(2)
            # tempt=iter(temp)
            # batch=tempt.next()
            # print(batch[1]["text"].shape)

            if args.num_att_plot != 0:
                plot_attention_iter_factory = cls.build_iter_factory(
                    args=args,
                    distributed_option=distributed_option,
                    mode="plot_att",
                )
            else:
                plot_attention_iter_factory = None

            # 8. Start training
            if args.use_wandb:
                if wandb is None:
                    raise RuntimeError("Please install wandb")

                try:
                    wandb.login()
                except wandb.errors.UsageError:
                    logging.info("wandb not configured! run `wandb login` to enable")
                    args.use_wandb = False

            if args.use_wandb:
                if (
                    not distributed_option.distributed
                    or distributed_option.dist_rank == 0
                ):
                    if args.wandb_project is None:
                        project = "ESPnet_" + cls.__name__
                    else:
                        project = args.wandb_project

                    if args.wandb_name is None:
                        name = str(Path(".").resolve()).replace("/", "_")
                    else:
                        name = args.wandb_name

                    wandb.init(
                        entity=args.wandb_entity,
                        project=project,
                        name=name,
                        dir=output_dir,
                        id=args.wandb_id,
                        resume="allow",
                    )
                    wandb.config.update(args)
                else:
                    # wandb also supports grouping for distributed training,
                    # but we only logs aggregated data,
                    # so it's enough to perform on rank0 node.
                    args.use_wandb = False

            # Don't give args to trainer.run() directly!!!
            # Instead of it, define "Options" object and build here.
            trainer_options = cls.trainer.build_options(args)
            trainer_options.ilmt_accum_grad=args.ilmt_accum_grad
            trainer_options.ilmt_loss_weight=args.ilmt_loss_weight
            cls.trainer.run(
                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                train_iter_factory=train_iter_factory,
                ilmt_iter_factory=ilmt_train_iter_factory,
                valid_iter_factory=valid_iter_factory,
                plot_attention_iter_factory=plot_attention_iter_factory,
                trainer_options=trainer_options,
                distributed_option=distributed_option,
            )

            if wandb.run:
                wandb.finish()

    trainer=Trainer_ilmt

    @classmethod
    def build_ilmt_iter_options(
        cls,
        args: argparse.Namespace,
        distributed_option,
        mode: str,
            num_iters_per_epoch
    ):
        if mode == "train":
            preprocess_fn = cls.build_preprocess_fn(args, train=True)
            collate_fn = cls.build_collate_fn(args, train=True)
            data_path_and_name_and_type = args.ilmt_lm_data
            shape_files = args.ilmt_lm_data_shape_file
            batch_size = args.batch_size   #这里不支持
            batch_bins = args.ilmt_batch_bins
            batch_type = args.batch_type
            max_cache_size = args.max_cache_size
            max_cache_fd = args.max_cache_fd
            distributed = distributed_option.distributed
            num_batches = None
            num_iters_per_epoch = num_iters_per_epoch
            train = True


        return IteratorOptions(
            preprocess_fn=preprocess_fn,
            collate_fn=collate_fn,
            data_path_and_name_and_type=data_path_and_name_and_type,
            shape_files=shape_files,
            batch_type=batch_type,
            batch_size=batch_size,
            batch_bins=batch_bins,
            num_batches=num_batches,
            max_cache_size=max_cache_size,
            max_cache_fd=max_cache_fd,
            distributed=distributed,
            num_iters_per_epoch=num_iters_per_epoch,
            train=train,
        )

