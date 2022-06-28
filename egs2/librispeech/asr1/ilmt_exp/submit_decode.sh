#!/bin/bash -x


config_file=$1
decode_epoch=$2
num_of_gpu=4

exps_dir=/home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/exp/downmodel/ilmt_exp


ls $config_file > /dev/null || exit 1;
exp_name="$(basename "${config_file}" .yaml)"

echo $exp_name

if [ ! -n "$decode_epoch" ]; then
  
newest_epoch="$(basename `ls -at $exps_dir/$exp_name/*epoch.pth | head -n 1`)"
echo decoding newest epoch: $newest_epoch

decode_epoch=$newest_epoch
  
else
  
decode_epoch=${decode_epoch}.pth
if [ ! -f $exps_dir/$exp_name/$decode_epoch ]; then

echo $exps_dir/$exp_name/$decode_epoch not found

#exit 1;
 echo WARNING==================WARNING 
else
echo decoding $decode_epoch
fi  
  

  
fi
if [ $num_of_gpu -ge 4 ]; then


wt="48:00:00"

else
wt="24:00:00"
fi



# echo qsub_decode_epoch=$decode_epoch,qsub_exp_dir=$exps_dir/$exp_name,qsub_num_of_gpu=$num_of_gpu
# exit 0;
qsub -P 12001458 -j oe -q dgx -l select=1:ncpus=5:ngpus=$num_of_gpu -N decode_$exp_name \
	-l walltime=$wt -o $exps_dir/$exp_name/qsub_decode.log \
	-v qsub_decode_epoch=$decode_epoch,qsub_exp_dir=$exps_dir/$exp_name,qsub_num_of_gpu=$num_of_gpu \
	./ilmt_exp/decode.sh    
 
 
	
	
	