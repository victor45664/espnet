#!/bin/bash
#victor 2020.9.12
cd /home/projects/12001458/yufei/espnet/egs2/voice_conversion/vc1
. ./path.sh

mutation=$qsub_mutation
ifresume_training=$qsub_ifresume_training

if [ $ifresume_training == false ]
then

mkdir -p $(dirname $0)/log
mkdir -p $(dirname $0)/newest_model_saved/$mutation


rm -rf $(dirname $0)"/log/"$mutation   #删除上次训练的记录


python  -u $(dirname $0)/train.py $mutation 0 || exit 1;


else

echo "resume training"
python  -u  $(dirname $0)/train.py $mutation 1 || exit 1;

fi
