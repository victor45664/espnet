#!/usr/bin/env bash
# Copyright 2012-2015  Johns Hopkins University (Author: Daniel Povey).
#  Apache 2.0.



# This script obtains phone posteriors from a trained chain model, using either
# the xent output or the forward-backward posteriors from the denominator fst.
# The phone posteriors will be in matrices where the column index can be
# interpreted as phone-index - 1.

# You may want to mess with the compression options.  Be careful: with the current
# settings, you might sometimes get exact zeros as the posterior values.

# CAUTION!  This script isn't very suitable for dumping features from recurrent
# architectures such as LSTMs, because it doesn't support setting the chunk size
# and left and right context.  (Those would have to be passed into nnet3-compute
# or nnet3-chain-compute-post).

# Begin configuration section.
stage=2

nj=1  # Number of jobs to run.
cmd=ppg_extractor/run.pl
remove_word_position_dependency=false
use_xent_output=false
online_ivector_dir=
use_gpu=false
count_smoothing=1.0  # this should be some small number, I don't think it's critical;
                     # it will mainly affect the probability we assign to phones that
                     # were never seen in training.  note: this is added to the raw
                     # transition-id occupation counts, so 1.0 means, add a single
                     # frame's count to each transition-id's counts.

# End configuration section.

set -e -u
echo "$0 $@"  # Print the command line for logging

[ -f kaldi_path.sh ] && . ./kaldi_path.sh
. ppg_extractor/parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: $0 <nnet3-tree-ali-dir> <nnet3-model-dir>  <data-dir> <phone-post-dir>"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (run.pl|queue.pl|... <queue opts>)    # how to run jobs."
  echo "  --config <config-file>                      # config containing options"
  echo "  --stage <stage>                             # stage to do partial re-run from."
  echo "  --nj <N>                                    # Number of parallel jobs to run, default:1"
  echo "  --remove-word-position-dependency <bool>    # If true, remove word-position-dependency"
  echo "                                              # info when dumping posteriors (default: false)"
  echo "  --use-xent-output <bool>                    # If true, use the cross-entropy output of the"
  echo "                                              # neural network when dumping posteriors"
  echo "                                              # (default: false, will use chain denominator FST)"
  echo "  --online-ivector-dir <dir>                  # Directory where we dumped online-computed"
  echo "                                              # ivectors corresponding to the data in <data>"
  echo "  --use-gpu <bool>                            # Set to true to use GPUs (not recommended as the"
  echo "                                              # binary is very poorly optimized for GPU use)."
  exit 1;
fi


tree_dir=$1
model_dir=$2
data=$3
dir=$4

for f in $model_dir/final.mdl $data/feats.scp; do
  [ ! -f $f ] && echo "get_bottleneck: no such file $f" && exit 1;
done

sdata=$data/split${nj}utt
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || ppg_extractor/split_data.sh --per-utt $data $nj || exit 1;

use_ivector=false

cmvn_opts=$(cat $model_dir/cmvn_opts)
feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"

if [ ! -z "$online_ivector_dir" ];then
  ppg_extractor/check_ivectors_compatible.sh $model_dir $online_ivector_dir || exit 1;
  ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
  ivector_feats="scp:ppg_extractor/filter_scp.pl $sdata/JOB/utt2spk $online_ivector_dir/ivector_online.scp |"
  ivector_opts="--online-ivector-period=$ivector_period --online-ivectors='$ivector_feats'"
else
  ivector_opts=
fi

if $use_gpu; then
  gpu_queue_opts="--gpu 1"
  gpu_opt="--use-gpu=yes"
  if ! cuda-compiled; then
    echo "$0: WARNING: you are running with one thread but you have not compiled"
    echo "   for CUDA.  You may be running a setup optimized for GPUs.  If you have"
    echo "   GPUs and have nvcc installed, go to src/ and do ./configure; make"
    exit 1
  fi
else
  gpu_queue_opts=
  gpu_opt="--use-gpu=no"
fi

mkdir -p $dir/log



if [ $stage -le 2 ]; then

  # note: --compression-method=3 is kTwoByteAuto: Each element is stored in two
  # bytes as a uint16, with the representable range of values chosen
  # automatically with the minimum and maximum elements of the matrix as its
  # edges.
  echo doing forward-prop ...
  compress_opts="--compress=true"
  model="nnet3-copy --nnet-config=${model_dir}/bottle_neck.config ${model_dir}/final.mdl -|"
  $cmd $gpu_queue_opts JOB=1:$nj $dir/log/get_phone_post.JOB.log \
       nnet3-compute $gpu_opt $ivector_opts \
       "$model"  "$feats" ark:- \| \
       copy-feats $compress_opts ark:- ark,scp:$dir/phone_post.JOB.ark,$dir/phone_post.JOB.scp || exit 1;

  sleep 5
  # Make a single .scp file, for convenience.
  for n in $(seq $nj); do cat $dir/phone_post.$n.scp; done > $dir/phone_post.scp
  echo done forward-prop ...
fi
