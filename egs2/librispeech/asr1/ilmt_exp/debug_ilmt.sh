#!/bin/bash
#PBS -P 12001458                                           
#PBS -j oe                                                    
#PBS -q dgx                                                 
#PBS -l select=1:ncpus=10:ngpus=1  
#PBS -l walltime=12:00:00
#PBS -o /home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/exp/downmodel/ilmt_exp/debug/qsub.log                                

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
exp_dir=/home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/exp/downmodel/ilmt_exp/debug

python3 -m espnet2.bin.asr_train_ilmt \
--ngpu 1 \
--use_preprocessor true \
--bpemodel data/en_token_list/bpe_unigram5000/bpe.model \
--token_type bpe \
--token_list data/en_token_list/bpe_unigram5000/tokens.txt \
--non_linguistic_symbols none \
--cleaner none \
--g2p none \
--valid_data_path_and_name_and_type dump/raw/dev/wav.scp,speech,sound \
--valid_data_path_and_name_and_type dump/raw/dev/text,text,text \
--valid_shape_file exp/asr_stats_raw_en_bpe5000_sp/valid/speech_shape \
--valid_shape_file exp/asr_stats_raw_en_bpe5000_sp/valid/text_shape.bpe \
--resume true \
--ignore_init_mismatch false \
--fold_length 80000 \
--fold_length 150 \
--output_dir $exp_dir \
--config /home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/ilmt_exp/conf/newtrain_fe.yaml \
--frontend_conf fs=16k \
--normalize=global_mvn \
--normalize_conf stats_file=/home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/exp/downmodel/asr_stats_raw_bpe5000_n_fft512_hop256_sp/train/feats_stats.npz \
--train_data_path_and_name_and_type dump/raw/train_960_sp/wav.scp,speech,sound \
--train_data_path_and_name_and_type dump/raw/train_960_sp/text,text,text \
--train_shape_file exp/asr_stats_raw_en_bpe5000_sp/train/speech_shape \
--train_shape_file exp/asr_stats_raw_en_bpe5000_sp/train/text_shape.bpe \
--log_interval 10


	
	
	
	