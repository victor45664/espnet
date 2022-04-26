# -*- coding: utf-8 -*-
import sys
sys.path.append('')
from espnet2.VC_SRC.VC_utils import dynamic_import_from_abs_path,path_to_model_save_path
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES']=''

mutation_path = './models/multi_spk_deploy/mutation/baseline_mvemb_larger_larger_16K.py'
output_dir='./exp/output_model/multi_spk_deploy_baseline_mvemb_larger_larger_16K'

os.makedirs(output_dir,exist_ok=True)



vc_model_mynn = dynamic_import_from_abs_path(mutation_path)


BN_feat_plh = tf.placeholder(tf.float32, [None, None, 512], 'BN_feat')  #手动调节
speaker_plh = tf.placeholder(tf.int32, [None], 'speaker')
seq_length_plh = tf.placeholder(tf.int32, [None], 'seq_length')
#speaker_plh=speaker_plh*0+8  #daji在第八个spk上，转换到0

melgen_outputs, final_outputs = vc_model_mynn.VC_convertion_model(BN_feat_plh, speaker_plh, False, None, seq_length=seq_length_plh,
                                                                  number_of_speaker=21)   #手动调节
sess=tf.Session()
saver=tf.train.Saver()
saver.restore(sess,
                 save_path=tf.train.latest_checkpoint(path_to_model_save_path(mutation_path)))

saver.save(sess, output_dir)







