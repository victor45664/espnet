#!/bin/bash

nvidia-smi                                        ##  <-- your script, like what you are run in local!
cd /home/projects/12001458/yufei
source /home/projects/12001458/yufei/path-dgx.sh
cd /home/projects/12001458/yufei/espnet/egs2/librispeech/asr1
. ./path.sh

set -e
set -u
set -x
set -o pipefail

param_set[0]="--lm_weight=0.5,0.8 --ctc_weight=0.3 --beam_size 60"  #shallow fusion 解码
param_set[1]="--lm_weight=0.6,0.7 --ctc_weight=0.3 --beam_size 60"
#param_set[2]="--ilm_weight -0.6 --lm_weight=0.7,0.6 --ctc_weight=0.3 --beam_size 60"  #ilme解码，用于对比

inference_asr_model=$qsub_decode_epoch
asr_exp=/home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/exp/downmodel/asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp_ilme_share  #虽然用了ilme的模型，但是这里不会使用ilme，这是为了方便与ilme对比效果
data_feats=/home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/dump/raw
dset=test_other
_data="${data_feats}/${dset}"
decode_dir=$qsub_exp_dir/decode_$inference_asr_model
_logdir=$decode_dir/logdir
bpemodel=/home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/data/en_token_list/bpe_unigram5000/bpe.model
python=python3
cleaner=none
nlsyms_txt=none			  	   
_nj=4
_opts=
_opts+="--lm_train_config /home/projects/12001458/yufei/espnet/egs2/librispeech/asr1/exp/downmodel/lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue/config.yaml "
_opts+="--lm_file $qsub_exp_dir/${inference_asr_model}_noilm "

python3 -m espnet2.bin.remove_ilm $qsub_exp_dir/${inference_asr_model}








mkdir -p "${_logdir}"

_feats_type="$(<${_data}/feats_type)"
if [ "${_feats_type}" = raw ]; then
	_scp=wav.scp
	_type=sound
else
	_scp=feats.scp
	_type=kaldi_ark
fi

# 1. Split the key file
key_file=${_data}/${_scp}
split_scps=""

for n in $(seq "${_nj}"); do
	split_scps+=" ${_logdir}/keys.${n}.scp"
done
# shellcheck disable=SC2086
utils/split_scp.pl "${key_file}" ${split_scps}

# 2. Submit decoding jobs
# shellcheck disable=SC2086
now=`date +'%Y-%m-%d %H:%M:%S'`
start_time=$(date --date="$now" +%s);
 
 
 for ((i = 0; i < ${#param_set[@]}; i++))
do
	c_opts=$_opts"${param_set[$i]}"
    run.pl JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
python3 -m espnet2.bin.asr_inference_ilme \
		--batch_size 1 \
		--ngpu 1 \
		--data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
		--key_file "${_logdir}"/keys.JOB.scp \
		--assigngpu '`expr \( JOB - 1 \) / 2`' \
		--asr_train_config $asr_exp/config.yaml \
		--asr_model_file $asr_exp/lmkd_exp_baseline.pth \
		--output_dir "${_logdir}"/output.JOB \
		${c_opts}
done
#你的脚本


now=`date +'%Y-%m-%d %H:%M:%S'`
end_time=$(date --date="$now" +%s);
echo "decode time:"$((end_time-start_time))"s"
					 
			 

mkdir -p ${_logdir}/transpose
for i in $(seq "${_nj}"); do
	for param_dir in $(ls ${_logdir}/output.${i});do
		if [ -d ${_logdir}/output.${i}/$param_dir ]; then
		
		
			mkdir -p ${_logdir}/transpose/$param_dir/output.${i}
			cp -a ${_logdir}/output.${i}/$param_dir/* ${_logdir}/transpose/$param_dir/output.${i}
		fi
	done  
done
						   
											  

set +e
for param_dir in $(ls ${_logdir}/transpose);do
mkdir -p $decode_dir/${param_dir}
																  
										   
				  
			
																	
										
										  
											   
														  
										
		  


												 
for f in token token_int score text; do
	if [ -f "${_logdir}/transpose/${param_dir}/output.1/1best_recog/${f}" ]; then
	  for i in $(seq "${_nj}"); do
		  cat "${_logdir}/transpose/${param_dir}/output.${i}/1best_recog/${f}"
	  done | sort -k1 > "${decode_dir}/${param_dir}/${f}"
	fi
done



            _data="${data_feats}/${dset}"
            _dir="${decode_dir}/${param_dir}"

            for _type in cer wer ter; do
                [ "${_type}" = ter ] && [ ! -f "${bpemodel}" ] && continue

                _scoredir="${_dir}/score_${_type}"
                mkdir -p "${_scoredir}"

                if [ "${_type}" = wer ]; then
                    # Tokenize text to word level
                    paste \
                        <(<"${_data}/text" \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type word \
                                  --non_linguistic_symbols "${nlsyms_txt}" \
                                  --remove_non_linguistic_symbols true \
                                  --cleaner "${cleaner}" \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/ref.trn"

                    # NOTE(kamo): Don't use cleaner for hyp
                    paste \
                        <(<"${_dir}/text"  \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type word \
                                  --non_linguistic_symbols "${nlsyms_txt}" \
                                  --remove_non_linguistic_symbols true \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/hyp.trn"


                elif [ "${_type}" = cer ]; then
                    # Tokenize text to char level
                    paste \
                        <(<"${_data}/text" \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type char \
                                  --non_linguistic_symbols "${nlsyms_txt}" \
                                  --remove_non_linguistic_symbols true \
                                  --cleaner "${cleaner}" \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/ref.trn"

                    # NOTE(kamo): Don't use cleaner for hyp
                    paste \
                        <(<"${_dir}/text"  \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type char \
                                  --non_linguistic_symbols "${nlsyms_txt}" \
                                  --remove_non_linguistic_symbols true \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/hyp.trn"

                elif [ "${_type}" = ter ]; then
                    # Tokenize text using BPE
                    paste \
                        <(<"${_data}/text" \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type bpe \
                                  --bpemodel "${bpemodel}" \
                                  --cleaner "${cleaner}" \
                                ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/ref.trn"

                    # NOTE(kamo): Don't use cleaner for hyp
                    paste \
                        <(<"${_dir}/text" \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type bpe \
                                  --bpemodel "${bpemodel}" \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/hyp.trn"

                fi

                sclite \
                    -r "${_scoredir}/ref.trn" trn \
                    -h "${_scoredir}/hyp.trn" trn \
                    -i rm -o all stdout > "${_scoredir}/result.txt"

                log "Write ${_type} result in ${_scoredir}/result.txt"
                grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
            done




done  
	
											 
	
	
