#!/bin/bash

nvidia-smi                                        ##  <-- your script, like what you are run in local!
cd /home/projects/12001458/yufei
source /home/projects/12001458/yufei/path-dgx.sh
cd /home/projects/12001458/yufei/espnet/egs2/librispeech/asr1
. ./path.sh

set -e
set -u
set -x
set -o pipefail

exp_name="$(basename "${qsub_config_file}" .yaml)"

exp_dir=/home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/exp/downmodel/lmkd_exp/$exp_name

python3 -m espnet2.bin.lm_train_kd \
--ngpu $qsub_numofgpu \
--use_preprocessor true \
--bpemodel data/en_token_list/bpe_unigram5000/bpe.model \
--token_type bpe \
--token_list data/en_token_list/bpe_unigram5000/tokens.txt \
--non_linguistic_symbols none \
--cleaner none \
--g2p none \
--valid_data_path_and_name_and_type dump/raw/dev/text,text,text \
--valid_shape_file exp/lm_stats_en_bpe5000/valid/text_shape.bpe \
--resume true \
--ignore_init_mismatch true \
--fold_length 150 \
--output_dir $exp_dir \
--config $qsub_config_file \
--train_data_path_and_name_and_type dump/raw/lm_train.txt,text,text \
--train_shape_file exp/lm_stats_en_bpe5000/train/text_shape.bpe \
--log_interval 100


	
	
	
	