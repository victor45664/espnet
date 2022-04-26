# -*- coding: utf-8 -*-
import sys
sys.path.append('')
from espnet2.VC_SRC.During_training import BNfeats_meltarget_spk_dataset
from torch.utils.data import DataLoader
from espnet2.VC_SRC.VC_utils import get_gpu
import kaldi_io
import tensorflow as tf
from espnet2.VC_SRC.evaluation.eval_dataset import eval_dataset
import os
import numpy as np
from espnet2.VC_SRC.Model_component import l2loss,l1_mel_loss
from espnet2.VC_SRC.VC_utils import path_to_module,path_to_model_save_path

mutation_path = './models/multi_spk_deploy_mel80/mutation/baseline_mvemb_larger_larger_finetune.py'
output_dir='./exp/extraction/multi_spk_deploy_mel80_baseline_mvemb_larger_larger_finetune'
os.makedirs(output_dir,exist_ok=True)


def down_sample_mel80(mel):
    #mel[B,l,d]
    mel[:,:,71:]=0
    return mel #VC模型输出的是24K的mel谱子，但ASR需要16K的mel谱，所以需要手动置 71-80为0

mynn = __import__(path_to_module(mutation_path,3), fromlist=True)



temp = mutation_path.split('/')
modelname = temp[-1]

model_save_dir = path_to_model_save_path(mutation_path)
print(model_save_dir)

train_dataset=BNfeats_meltarget_spk_dataset(os.path.join(mynn.finetune_datadir,'train'), output_per_step=mynn.tactron_hp.outputs_per_step, max_length_allowed=4000)  #用模型把训练集的mel谱都提取出来
train_loader=DataLoader(dataset=train_dataset,
                                  batch_size=mynn.training_hp.batchsize,
                                  shuffle=False,
                                  num_workers=4,
                                  drop_last=False,
                                    collate_fn=train_dataset.collect_fn)



eval_dataset=eval_dataset(os.path.join(mynn.finetune_datadir,'train'), output_per_step=mynn.tactron_hp.outputs_per_step)
eval_loader=DataLoader(dataset=eval_dataset,
                                  batch_size=mynn.training_hp.batchsize,
                                  shuffle=False,
                                  num_workers=4,
                                  drop_last=False,
                                    collate_fn=eval_dataset.collect_fn)



BN_feat_plh= tf.placeholder(tf.float32, [None, None, train_dataset.BN_feat_dim], 'BN_feat')

speaker_plh=tf.placeholder(tf.int32, [None], 'speaker')
seq_length_plh=tf.placeholder(tf.int32, [None], 'seq_length')
mel_target_plh= tf.placeholder(tf.float32, [None, None, train_dataset.mel_target_dim], 'mel_target')



melgen_outputs,final_outputs=mynn.VC_convertion_model(BN_feat_plh, speaker_plh, False, mel_target_plh, number_of_speaker=19, seq_length=seq_length_plh,name_space="VC_convertion_model")



final_outputs_loss=l1_mel_loss(final_outputs, mel_target_plh)
total_loss= l1_mel_loss(melgen_outputs, mel_target_plh) + l2loss() + final_outputs_loss



asr_model_path=r'$root_path/tf_project/tensorflow_ASR/models/ASR_mel80/newest_model_saved/baseline/baseline-282500'

_ = tf.train.import_meta_graph(asr_model_path + '.meta', clear_devices=True)


gpu_list=get_gpu(1)
if len(gpu_list)==0:
    sys.exit(1)
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
vc_var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='VC_convertion_model')+[sess.graph.get_tensor_by_name('global_step:0')]
saver_newest = tf.train.Saver(max_to_keep=1,var_list=vc_var_list)
model_file=tf.train.latest_checkpoint(model_save_dir)
print(model_file)
saver_newest.restore(sess,model_file)

asr_input_phl=sess.graph.get_tensor_by_name('input_MFCC:0')
asr_BN_feat_op=sess.graph.get_tensor_by_name('ASR_model/ASR_model/TDNN7/TDNN7/Relu:0')
is_training_plh = sess.graph.get_tensor_by_name('is_training:0')
asr_model_saver = tf.train.Saver(
    var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ASR_model'), max_to_keep=3)
asr_model_saver.restore(sess, asr_model_path) #应该先初始化再restore


train_loader_iter=iter(train_loader)
eval_loader_iter=iter(eval_loader)
ark_files=[]
for spk in range(0,19):
    ark_files.append(open(os.path.join(output_dir, 'spk_'+str(spk)+'.ark'), 'wb'))

while(1):
    try:
        BN_feat_train, mel_target_train, original_spk, BN_feat_seq_length=train_loader_iter.next()
        uttids, BN_feat_test, BN_feat_seq_length_eval=eval_loader_iter.next()

    except StopIteration:
        break

    for spk in range(0,19):


        target_spk= np.ones((BN_feat_train.shape[0]), dtype=int)*spk #先变成其它的spk


        temp = sess.run([total_loss, final_outputs_loss,final_outputs],
                        feed_dict={BN_feat_plh: BN_feat_train, mel_target_plh: mel_target_train, speaker_plh: target_spk, seq_length_plh:BN_feat_seq_length})
        self_vc_loss = temp[1]                   #当finetune数据集有多个spk的时候，这里的mel_target_train其实是错的，因此不需要在意这里计算出来的loss
                                                  #但如果只有一个spk的时候这里的loss是正确的

        other_spk_mel=temp[2]

        other_spk_mel=down_sample_mel80(other_spk_mel) #这里输出的是24K的mel谱，需要手动截断成16K的mel谱
        BN_feat_from_other_spk = sess.run(asr_BN_feat_op,
                           feed_dict={asr_input_phl: other_spk_mel,
                                      is_training_plh: False})  #来自其它spk的BN特征

        temp = sess.run([total_loss, final_outputs_loss,final_outputs],
                        feed_dict={BN_feat_plh: BN_feat_from_other_spk, mel_target_plh: mel_target_train, speaker_plh: original_spk, seq_length_plh:BN_feat_seq_length})

        circle_vc_loss = temp[1]

        print("extracting spk {} self vc loss:{} circle vc loss:{}".format(str(spk),self_vc_loss,circle_vc_loss))
        mels=temp[2]
        for i in range(len(uttids)):
            key=uttids[i]
            length=BN_feat_seq_length[i]
            kaldi_io.write_mat(ark_files[spk], mels[i,0:length,:], key=key+"_"+str(spk))

