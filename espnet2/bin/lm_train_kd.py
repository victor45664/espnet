#!/usr/bin/env python3
from espnet2.tasks.lm import LMTask_kd


def get_parser():
    parser = LMTask_kd.get_parser()
    return parser


def main(cmd=None):
    """LM training.

    Example:

        % python lm_train.py asr --print_config --optim adadelta
        % python lm_train.py --config conf/train_asr.yaml
    """
    LMTask_kd.main(cmd=cmd)


if __name__ == "__main__":
    main()
