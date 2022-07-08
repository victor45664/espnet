#!/bin/bash
#victor 2022.4.12


nvidia-smi                                        ##  <-- your script, like what you are run in local!
cd /home/projects/12001458/yufei
source /home/projects/12001458/yufei/path-dgx.sh
cd /home/projects/12001458/yufei/espnet/egs2/voice_conversion/vc1
. ./path.sh

cd ./$qsub_modelname
ls

./train.sh $qsub_mutation  $qsub_ifresume_training
