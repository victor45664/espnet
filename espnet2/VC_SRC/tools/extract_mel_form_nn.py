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
from espnet2.VC_SRC.Model_component import l2loss,l1_mel_loss
from espnet2.VC_SRC.VC_utils import path_to_module,path_to_model_save_path

mutation_path = './models/multi_spk_deploy/mutation/baseline_mvemb_larger_larger_24K_finetune.py'
output_dir='./exp/extraction/multi_spk_deploy_baseline_mvemb_larger_larger_24K_finetune'
os.makedirs(output_dir,exist_ok=True)


mynn = __import__(path_to_module(mutation_path,3), fromlist=True)



temp = mutation_path.split('/')
modelname = temp[-1]

model_save_dir = path_to_model_save_path(mutation_path)
print(model_save_dir)

train_dataset=BNfeats_meltarget_spk_dataset(os.path.join(mynn.datadir,'train'), output_per_step=mynn.tactron_hp.outputs_per_step, max_length_allowed=4000)  #用模型把训练集的mel谱都提取出来
train_loader=DataLoader(dataset=train_dataset,
                                  batch_size=mynn.training_hp.batchsize,
                                  shuffle=False,
                                  num_workers=4,
                                  drop_last=False,
                                    collate_fn=train_dataset.collect_fn)



eval_dataset=eval_dataset(os.path.join(mynn.datadir,'train'), output_per_step=mynn.tactron_hp.outputs_per_step)
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





gpu_list=get_gpu(1)
if len(gpu_list)==0:
    sys.exit(1)
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
saver_newest = tf.train.Saver(max_to_keep=1)
model_file=tf.train.latest_checkpoint(model_save_dir)
print(model_file)
saver_newest.restore(sess,model_file)




train_loader_iter=iter(train_loader)
eval_loader_iter=iter(eval_loader)
with open(os.path.join(output_dir, 'out.ark'), 'wb') as f:
    while(1):
        try:
            BN_feat_train,mel_target_train,spk_train,BN_feat_seq_length=train_loader_iter.next()
            uttids, BN_feat_test, BN_feat_seq_length_eval=eval_loader_iter.next()

        except StopIteration:
            break
       # spk_train=spk_train*0+8
        temp = sess.run([total_loss, final_outputs_loss,final_outputs],
                        feed_dict={BN_feat_plh: BN_feat_train, mel_target_plh: mel_target_train, speaker_plh: spk_train,seq_length_plh:BN_feat_seq_length})
        print("loss on training set:","total_loss",temp[0],"mel_loss",temp[1])#这里用来确保模型被正确的加载
        mels=temp[2]
        for i in range(len(uttids)):
            key=uttids[i]
            length=BN_feat_seq_length[i]
            kaldi_io.write_mat(f, mels[i,0:length,:], key=key)

