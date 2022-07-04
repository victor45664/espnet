#!/bin/bash
#PBS -P 12001458                                           
#PBS -j oe                                                    
#PBS -q dgx                                                 
#PBS -l select=1:ncpus=10:ngpus=1  
#PBS -l walltime=12:00:00
#PBS -o /home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/exp/downmodel/lmkd_exp/debug/qsub.log                                

nvidia-smi                                        ##  <-- your script, like what you are run in local!
cd /home/projects/12001458/yufei
source /home/projects/12001458/yufei/path-dgx.sh
cd /home/projects/12001458/yufei/espnet/egs2/librispeech/asr1
. ./path.sh
export CUDA_VISIBLE_DEVICES=1
set -e
set -u
set -x
set -o pipefail
exp_dir=/home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/exp/downmodel/lmkd_exp/debug

python3 -m espnet2.bin.lm_train_kd \
--ngpu 1 \
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
--config lmkd_exp/conf/new_lmkd_debug.yaml \
--train_data_path_and_name_and_type dump/raw/lm_train.txt,text,text \
--train_shape_file exp/lm_stats_en_bpe5000/train/text_shape.bpe \
--log_interval 100


	
	
	
	