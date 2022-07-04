#!/bin/bash
#PBS -P 12001458                                           
#PBS -j oe                                                    
#PBS -q dgx                                                 
#PBS -l select=1:ncpus=4:ngpus=2  
#PBS -l walltime=24:00:00
#PBS -o /home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/lmkd_exp/run_lmkd_oracle3.log                                

nvidia-smi                                        ##  <-- your script, like what you are run in local!
cd /home/projects/12001458/yufei
source /home/projects/12001458/yufei/path-dgx.sh
cd /home/projects/12001458/yufei/espnet/egs2/librispeech/asr1
. ./path.sh
echo "vdebug" oracle3 $CUDA_VISIBLE_DEVICES
set -e
set -u
set -x
set -o pipefail
#export CUDA_VISIBLE_DEVICES=2,3
qsub_numofgpu=2
qsub_config_file=/home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/lmkd_exp/conf/oracle_0.6.yaml

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
--train_data_path_and_name_and_type /home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/exp/downmodel/asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp_ilme_share/decode_2epoch.pth_extract/all_hyp_uniq,text,text \
--train_shape_file /home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/exp/downmodel/asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp_ilme_share/decode_2epoch.pth_extract/train/text_shape.bpe \
--log_interval 1000000


	
	
	
	