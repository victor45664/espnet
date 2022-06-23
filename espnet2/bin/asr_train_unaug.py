#!/usr/bin/env python3
from espnet2.tasks.asr2 import ASRTask_unaug as ASRTask
#对于beamsearch中出错的候选做专门的 训练

def get_parser():
    parser = ASRTask.get_parser()
    return parser


def main(cmd=None):
    r"""ASR training.

    Example:

        % python asr_train.py asr --print_config --optim adadelta \
                > conf/train_asr.yaml
        % python asr_train.py --config conf/train_asr.yaml
    """
    ASRTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
