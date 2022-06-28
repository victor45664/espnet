#!/bin/bash 


config_file=$1
decode_epoch=$2
num_of_gpu=2

exps_dir=/home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/exp/downmodel/lmkd_exp


ls $config_file > /dev/null || exit 1;
exp_name="$(basename "${config_file}" .yaml)"

echo $exp_name

if [ ! -n "$decode_epoch" ]; then
  
newest_epoch="$(basename `ls -at $exps_dir/$exp_name/*epoch.pth | head -n 1`)"
echo decoding newest epoch: $newest_epoch

decode_epoch=$newest_epoch
  
else
  
decode_epoch=$decode_epoch
if [ ! -f $exps_dir/$exp_name/$decode_epoch ]; then

echo $exps_dir/$exp_name/$decode_epoch not found

exit 1;
  
else
echo decoding $decode_epoch
fi  
  

  
fi


if [[ "$HOSTNAME" =~ "nus" ]]
then

qsub -P 12001458 -j oe -q dgx -l select=1:ncpus=2:ngpus=$num_of_gpu -N decode_$exp_name \
	-l walltime=24:00:00 -o $exps_dir/$exp_name/qsub_decode.log \
	-v qsub_decode_epoch=$decode_epoch,qsub_exp_dir=$exps_dir/$exp_name \
	./lmkd_exp/decode.sh 


else


export qsub_decode_epoch=$decode_epoch;
export qsub_exp_dir=$exps_dir/$exp_name;
export qsub_num_of_gpu=$num_of_gpu;

	./lmkd_exp/decode.sh    
fi
exit 0
# echo qsub_decode_epoch=$decode_epoch,qsub_exp_dir=$exps_dir/$exp_name
# exit 0;
qsub -P 12001458 -j oe -q dgx -l select=1:ncpus=2:ngpus=$num_of_gpu -N decode_$exp_name \
	-l walltime=24:00:00 -o $exps_dir/$exp_name/qsub_decode.log \
	-v qsub_decode_epoch=$decode_epoch,qsub_exp_dir=$exps_dir/$exp_name \
	./lmkd_exp/decode.sh    

	
	
	