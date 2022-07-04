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
qsub_numofgpu=0
qsub_config_file=/home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/lmkd_exp/conf/new_lmkd_4gpu_0.5_nols.yaml

exp_name="$(basename "${qsub_config_file}" .yaml)"

exp_dir=/home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/exp/lm_stats_en_bpe5000/test_other

python3 -m espnet2.bin.lm_train_kd \
--collect_stats true \
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
--train_data_path_and_name_and_type /home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/dump/raw/test_other/text,text,text \
--log_interval 100


	
	
	
	