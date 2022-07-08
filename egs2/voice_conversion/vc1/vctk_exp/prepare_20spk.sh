#!/bin/bash -x
#victor 2020.9.12
#使用训练好的kaldi模型提取BN特征，用于VC模型的训练
#注意，这个脚本仅需要运行一次，之后所有mutation都可以训练和eval
#set -e

KALDI_ROOT=/home/projects/12001458/yizhoupeng/w2021/kaldi-2021    
corpus_path=/home/projects/12001458/yufei/Dataset/VCTK_old/VCTK-Corpus/wav48    #VCTK数据集的路径

kaldi_export_path=/home/projects/12001458/yufei/kaldi/librispeech/export_model         #打包好的kaldi预训练模型
ppg_tool=/home/users/nus/e0934250/espnet/espnet2/VC_SRC/preprocessing/kaldi_util   #kaldi 工具箱  

PYTHON=/home/projects/12001458/yufei/anaconda3/bin/python    #python interpreter


current_dir=$(dirname $0)
data_dir=$current_dir/data_20spk   #输出的文件夹


stage=0





#=============目标说话人的id
corpus_id="
p335
p252
p341
p256
p298
p310
p281
p259
p276
p274
p329
p247
p297
p295
p316
p271
p323
p243
p265
p308
"

#p335  25  F    NewZealand  English
#p252  22  M    Scottish  Edinburgh
#p341  26  F    American  Ohio
#p256  24  M    English    Birmingham
#p298  19  M    Irish  Tipperary
#p310  21  F    American  Tennessee
#p281  29  M    Scottish  Edinburgh
#p259  23  M    English    Nottingham
#p276  24  F    English    Oxford
#p274  22  M    English    Essex
#p329  23  F    American
#p247  22  M    Scottish  Argyll
#p297  20  F    American  New  York
#p295  23  F    Irish  Dublin
#p316  20  M    Canadian  Alberta
#p271  19  M    Scottish  Fife
#p323  19  F    SouthAfrican  Pretoria
#p243  22  M    English    London
#p265  23  F    Scottish  Ross
#p308  18  F    American  Alabama

#=============用于eval时作为source的说话人的id
eval_corpus_id="
p336
p339
p340
p341
p343
p345
p347
p351
p360
p361
p362
p363
p364
p374
p376
"


[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C




corpus_names=`echo $corpus_id | sed 's/\n/ /g'`

if [ $stage -le 0 ]; then
#整理wav.scp
mkdir -p $data_dir/train  #用于训练的句子
mkdir -p $data_dir/dev   #存放模型没见过的上述speaker的句子，用来检查训了的时候的mel误差

rm -f $data_dir/train/wav.scp.tmp
rm -f $data_dir/dev/wav.scp.tmp

rm -f $data_dir/train/utt2spk_id.tmp
rm -f $data_dir/dev/utt2spk_id.tmp



spk_id=0
for corpus in $corpus_names;do   #准备训练数据
    wave_path=$corpus_path/$corpus/
    if [ ! -d "$wave_path" ]; then
      echo "Directory $wave_path DOES NOT EXIST." && exit 1
    fi
   find $wave_path -type f -name "*.wav" | awk -F"/" '{print $NF " " $0}' | sed 's/.wav / /g' | shuf | head -n 1100 > $data_dir/${corpus}_wav.scp


	tail -n 10 $data_dir/${corpus}_wav.scp >> $data_dir/dev/wav.scp.tmp
	tail -n 10 $data_dir/${corpus}_wav.scp | awk -v spk="$spk_id" '{print $1,spk}' >> $data_dir/dev/utt2spk_id.tmp


	head -n -10 $data_dir/${corpus}_wav.scp >> $data_dir/train/wav.scp.tmp
	head -n -10 $data_dir/${corpus}_wav.scp | awk -v spk="$spk_id" '{print $1,spk}' >> $data_dir/train/utt2spk_id.tmp
   #每个spk只提取10句作为dev集，其它的都用来训练
	 ((spk_id++))
done

cat $data_dir/train/wav.scp.tmp | sort -u > $data_dir/train/wav.scp
cat $data_dir/dev/wav.scp.tmp | sort -u > $data_dir/dev/wav.scp

cat $data_dir/train/utt2spk_id.tmp | sort -u > $data_dir/train/utt2spk_id  #这是给tensorflow 用的。
cat $data_dir/dev/utt2spk_id.tmp | sort -u > $data_dir/dev/utt2spk_id


cat $data_dir/train/wav.scp | awk '{print $1,$1}' > $data_dir/train/utt2spk
cat $data_dir/dev/wav.scp | awk '{print $1,$1}' > $data_dir/dev/utt2spk  #这是给kaldi用的，因为计算cmvn的时候需要per utt计算

cp $data_dir/train/utt2spk $data_dir/train/spk2utt
cp $data_dir/dev/utt2spk $data_dir/dev/spk2utt

rm -f $data_dir/train/utt2spk_id.tmp
rm -f $data_dir/dev/utt2spk_id.tmp
rm -f $data_dir/train/wav.scp.tmp
rm -f $data_dir/dev/wav.scp.tmp

for dataset in train dev;do

mv $data_dir/$dataset/wav.scp $data_dir/$dataset/wav_mel.scp
awk '{print $1,"sox",$2,"-r 16000 -t wav - |"}' $data_dir/$dataset/wav_mel.scp > $data_dir/$dataset/wav.scp

done

fi



corpus_names=`echo $eval_corpus_id | sed 's/\n/ /g'`



if [ $stage -le 1 ]; then #准备eval数据
mkdir -p $data_dir/eval
rm -f $data_dir/eval/wav.scp.tmp
rm -f $data_dir/eval/wav.scp

for corpus in $corpus_names;do
    wave_path=$corpus_path/$corpus/

    find $wave_path -type f -name "*.wav" | awk -F"/" '{print $NF " " $0}' | sed 's/.wav / /g' | sort -u |  head -n 20 >> $data_dir/eval/wav.scp.tmp


	#需要降采样，以计算MFCC和pitch
	#utt_id sox -r 16000 -t wav $wav_path -t wav - |
   #每个spk只提取10句作为dev集，其它的都用来训练
	 ((spk_id++))
done
cat $data_dir/eval/wav.scp.tmp | sort -u > $data_dir/eval/wav.scp
cat $data_dir/eval/wav.scp | awk '{print $1,$1}' > $data_dir/eval/utt2spk
cp $data_dir/eval/utt2spk $data_dir/eval/spk2utt


fi




nj=20

	
if [ $stage -le 2 ]; then  #提取feat.scp

ln -s  $ppg_tool ./ppg_extractor


for dataset in train dev eval;do

dir=$data_dir/$dataset

  $ppg_tool/make_mfcc.sh --cmd $ppg_tool/run.pl --nj $nj \
     --mfcc-config $kaldi_export_path/conf/mfcc_hires.conf \
     $dir $dir/log/make_mfcc/ $dir/mfcc

  $ppg_tool/compute_cmvn_stats.sh $dir $dir/log/make_mfcc $dir/mfcc 2>/dev/null




done

rm -f ./ppg_extractor

fi

if [ $stage -le 3 ]; then  #计算ivector特征
    echo "ivector"
ln -s  $ppg_tool ./ppg_extractor

for dataset in train dev eval;do
	dir=$data_dir/$dataset

  $ppg_tool/extract_ivectors_online.sh --cmd $ppg_tool/run.pl --nj $nj $dir $kaldi_export_path/extractor \
    $dir/ivectors

	done
rm -f ./ppg_extractor
fi

if [ $stage -le 4 ]; then  #计算BN特征

ln -s  $ppg_tool ./ppg_extractor
    echo "ppg model type: bottleneck"
for dataset in train dev eval;do

	dir=$data_dir/$dataset
	
    $ppg_tool/get_phone_post_nnet3_bottleneck.sh --use-gpu false --cmd $ppg_tool/run.pl --nj $nj  --online-ivector-dir $dir/ivectors \
      $kaldi_export_path/tree $kaldi_export_path/am_model $dir \
        $dir/ppg_phonetone

done


rm -f ./ppg_extractor


fi


if [ $stage -le 5 ]; then  #计算target_mel
echo "cal mel target"

for dataset in train dev;do
	dir=$data_dir/$dataset
	$PYTHON ../../../../espnet2/VC_SRC/preprocessing/cal_target_mel_from_wav_mel.py $dir
	
	
done

wait

fi























