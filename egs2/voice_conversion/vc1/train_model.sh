#!/bin/bash -x

#victor 2022.4.12

#example: ./train.sh ./exp1/mutation/bsl.py



str=$1





#num_of_gpu=$(grep "numofGPU=" $str | tr -cd "[0-9]")
num_of_gpu=1




array=(${str//// })



lenarray=${#array[*]}
mutation=${array[lenarray-1]}
mutarray=(${mutation//./ })
mutation=${mutarray[0]}

modelname=${array[lenarray-3]}

for i in $(seq 0 $[$lenarray-3])
do

modeldir=$modeldir${array[$i]}/
done




jobname=train_$mutation



ifresume_training=1



# echo qsub_mutation=$mutation,qsub_ifresume_training=$ifresume_training
# echo $modelname
# exit 0;

mkdir -p log/$modelname



if [[ "$HOSTNAME" =~ "nus" ]]
then



qsub  -P 12001458 -j oe -q dgx -l select=1:ncpus=1:ngpus=$num_of_gpu -N $jobname \
	-l walltime=24:00:00 -o log/$modelname/$jobname.log \
	-v qsub_mutation=$mutation,qsub_ifresume_training=$ifresume_training,qsub_modelname=$modelname \
	./local/run.sh


echo "begin to run "$modelname "mutation:"$mutation with $num_of_gpu GPU



else


#$modeldir"train.sh" $mutation $ifresume_training

export qsub_mutation=$mutation;export qsub_ifresume_training=$ifresume_training;export qsub_modelname=$modelname;./local/run.sh 2>&1 | tee log/$modelname/$jobname.log

fi






