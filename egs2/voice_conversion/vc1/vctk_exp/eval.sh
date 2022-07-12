#!/bin/bash
#victor 2020.9.12

mutation=$1

vocoder_dir=/home/projects/12001458/yufei/espnet/espnet2/VC_SRC/vocoder


python -u $(dirname $0)/eval.py $mutation 0 || exit 1;




python $vocoder_dir/vocoder.py --path_input_mels $(dirname $0)/eval_results/$mutation \
	 --checkpoint_file $vocoder_dir/vocoder \
	  --config_file $vocoder_dir/config.json
