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
stage=0

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

if [ $# != 5 ]; then
  echo "Usage: $0 <nnet3-tree-ali-dir> <nnet3-model-dir> <lang-dir> <data-dir> <phone-post-dir>"
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
lang=$3
data=$4
dir=$5


for f in $tree_dir/tree $tree_dir/final.mdl $tree_dir/ali.1.gz $tree_dir/num_jobs \
         $model_dir/final.mdl $data/feats.scp $lang/phones.txt; do
  [ ! -f $f ] && echo "train_sat.sh: no such file $f" && exit 1;
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

if [ $stage -le 0 ]; then
  if [ ! -f $dir/tacc ] || [ $dir/tacc -ot $tree_dir/ali.1.gz ]; then
    echo "$0: obtaining transition-id counts in $dir/tacc"
    # Obtain counts for each transition-id, from the alignments.
    this_nj=$(cat $tree_dir/num_jobs)
    $cmd JOB=1:$this_nj $dir/log/acc_taccs.JOB.log \
       ali-to-post "ark:gunzip -c $tree_dir/ali.JOB.gz|" ark:- \| \
       post-to-tacc $tree_dir/final.mdl ark:- $dir/tacc.JOB

    input_taccs=$(for n in $(seq $this_nj); do echo $dir/tacc.$n; done)
    $cmd $dir/log/sum_taccs.log \
         vector-sum --binary=false $input_taccs $dir/tacc

    rm $dir/tacc.*
  else
    echo "$0: skipping creation of $dir/tacc since it already exists."
  fi
fi


if [ $stage -le 1 ] && $remove_word_position_dependency; then
  echo "$0: creating $dir/phone_map.int"
  ppg_extractor/get_word_position_phone_map.pl $lang $dir
else
  # Either way, $dir/phones.txt will be a symbol table for the phones that
  # we are dumping (although the matrices we dump won't contain anything
  # for symbol 0 which is <eps>).
  grep -v '^#' $lang/phones.txt > $dir/phones.txt
fi

if [ $stage -le 1 ]; then
  # we want the phones in integer form as it's safer for processing by script.
  # $data/fake_phones.txt will just contain e.g. "0 0\n1 1\n....", it's used
  # to force show-transitions to print the phones as integers.
  awk '{print $2,$2}' <$lang/phones.txt >$dir/fake_phones.txt


  # The format of the 'show-transitions' command below is like the following:
  #show-transitions tempdir/phone_map.int exp/chain/tree_a_sp/final.mdl
  #Transition-state 1: phone = 1 hmm-state = 0 forward-pdf = 0 self-loop-pdf = 51
  # Transition-id = 1 p = 0.5 [self-loop]
  # Transition-id = 2 p = 0.5 [0 -> 1]
  #Transition-state 2: phone = 10 hmm-state = 0 forward-pdf = 0 self-loop-pdf = 51
  # Transition-id = 3 p = 0.5 [self-loop]
  # Transition-id = 4 p = 0.5 [0 -> 1]

  # The following inline script processes that info about the transition model
  # into the file $dir/phones_and_pdfs.txt, which has a line for each transition-id
  # (starting from number 1), and the format of each line is
  # <phone-id> <pdf-id>
  show-transitions $dir/fake_phones.txt $tree_dir/final.mdl | \
    perl -ane ' if(m/Transition-state.* phone = (\d+) .* pdf = (\d+)/) { $phone = $1; $pdf = $2; }
        if(m/Transition-id/) {  if (m/self-loop/) { print "$phone $pdf\n"; }
            else { print "$phone $pdf\n" } } ' > $dir/phones_and_pdfs.txt || exit 1;
  echo test
  # The following command just separates the 'tacc' file into a similar format
  # to $dir/phones_and_pdfs.txt, with one count per line, and a line per transition-id
  # starting from number 1.  We skip the first two fields which are "[ 0" (the 0 is
  # for transition-id=0, since transition-ids are 1-based), and the last field which is "]".
  awk '{ for (n=3;n<NF;n++) print $n; }' <$dir/tacc  >$dir/transition_counts.txt || exit 1;

  num_lines1=$(wc -l <$dir/phones_and_pdfs.txt)
  num_lines2=$(wc -l <$dir/transition_counts.txt)
  if [ $num_lines1 -ne $num_lines2 ]; then
    echo "$0: mismatch in num-lines between phones_and_pdfs.txt and transition_counts.txt: $num_lines1 vs $num_lines2"
    exit 1
  fi

  # after 'paste', the format of the data will be
  # <phone-id> <pdf-id> <data-count>
  # we add the count smoothing at this point.
  paste $dir/phones_and_pdfs.txt $dir/transition_counts.txt | \
     awk -v s=$count_smoothing '{print $1, $2, (s+$3);}' > $dir/combined_info.txt

  if $remove_word_position_dependency; then
    # map the phones to word-position-independent phones; you can see $dir/phones.txt
    # to interpret the final output.
    ppg_extractor/apply_map.pl -f 1 $dir/phone_map.int <$dir/combined_info.txt > $dir/temp.txt
    mv $dir/temp.txt $dir/combined_info.txt
  fi

  awk 'BEGIN{num_phones=1;num_pdfs=1;} { phone=$1; pdf=$2; count=$3; pdf_count[pdf] += count; counts[pdf,phone] += count;
       if (phone>num_phones) num_phones=phone; if (pdf>=num_pdfs) num_pdfs = pdf + 1; }
       END{ print "[ "; for(phone=1;phone<=num_phones;phone++) {
          for (pdf=0;pdf<num_pdfs;pdf++) printf("%.10f ", counts[pdf,phone]/pdf_count[pdf]);
           print ""; } print "]"; }' <$dir/combined_info.txt >$dir/transform.mat  #0-3479到1-130

fi


if [ $stage -le 2 ]; then

  # note: --compression-method=3 is kTwoByteAuto: Each element is stored in two
  # bytes as a uint16, with the representable range of values chosen
  # automatically with the minimum and maximum elements of the matrix as its
  # edges.
  echo doing forward-prop ...
  compress_opts="--compress=true"
  $cmd $gpu_queue_opts JOB=1:$nj $dir/log/get_phone_post.JOB.log \
       nnet3-compute $gpu_opt $ivector_opts \
       --apply-exp=true \
       $model_dir/final.mdl "$feats" ark:- \| \
       transform-feats $dir/transform.mat ark:- ark:- \| \
       copy-feats $compress_opts ark:- ark,scp:$dir/phone_post.JOB.ark,$dir/phone_post.JOB.scp

  sleep 5
  # Make a single .scp file, for convenience.
  for n in $(seq $nj); do cat $dir/phone_post.$n.scp; done > $dir/phone_post.scp
  echo done forward-prop ...
fi
