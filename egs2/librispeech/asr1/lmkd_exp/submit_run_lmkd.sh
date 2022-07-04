#!/bin/bash


config_file=$1


ls $config_file > /dev/null || exit 1;

num_of_gpu=$(grep "#numofgpu: " $config_file | tr -cd "[0-9]") 

if [ -z "${num_of_gpu}" ]; then
  

num_of_gpu=8
  
fi


if [ $num_of_gpu -ge 4 ]; then


wt="24:00:00"

else
wt="48:00:00"
fi


exp_name="$(basename "${config_file}" .yaml)"
echo $exp_name
mkdir -p /home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/exp/downmodel/lmkd_exp/$exp_name

qsub -P 12001458 -j oe -q dgx -l select=1:ncpus=8:ngpus=$num_of_gpu -N $exp_name \
	-l walltime=$wt -o /home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/exp/downmodel/lmkd_exp/$exp_name/qsub.log -v qsub_config_file=$config_file,qsub_numofgpu=$num_of_gpu ./lmkd_exp/run_lmkd.sh    
 
 
	
	
	