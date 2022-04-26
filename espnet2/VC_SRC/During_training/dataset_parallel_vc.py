# -*- coding: utf-8 -*-
import sys
sys.path.append('')
from torch.utils.data import Dataset
import time
import pandas as pd
import os
import threading
import numpy as np
import random
import kaldi_io
import math

def change_length(input_feat,disired_length):
    current_length=input_feat.shape[0]
    if current_length==disired_length:
        return input_feat
    elif current_length>disired_length:
        return input_feat[:disired_length,:]
    else:
        return np.pad(input_feat, ((0,disired_length-current_length), (0,0)), mode='constant')






def lcm(x, y):#最小公倍数
    #  获取最大的数
    if x > y:
        greater = x
    else:
        greater = y

    while (True):
        if ((greater % x == 0) and (greater % y == 0)):
            lcm = greater
            break
        greater += 1

    return lcm


def _round_up(length, multiple):
  remainder = length % multiple
  return length if remainder == 0 else length - remainder






class RA_parallel_Reader_General(object):
    def __init__(self, source_utt_scp_file_path, target_utt_scp_file_path):
        source_utt_scp=pd.read_csv(target_utt_scp_file_path, header=None, delim_whitespace=True, names=['key', 'source_utt'])
        target_utt_scp=pd.read_csv(source_utt_scp_file_path, header=None, delim_whitespace=True, names=['key', 'target_utt'])
        print("find {} source_utt".format(len(source_utt_scp)))
        print("find {} target_utt".format(len(target_utt_scp)))
        total_training_sample=pd.merge(source_utt_scp,target_utt_scp,how='outer',on='key')
        self.total_valid_sample=total_training_sample.dropna(axis=0, how='any')  #valid sample 是指两个都有数据的样本
        self.total_valid_sample=self.total_valid_sample.reset_index(drop=True)
        self.num_of_valid_sample=len(self.total_valid_sample)
        print("{} valid sample".format(self.num_of_valid_sample))


    def __len__(self):
        return self.num_of_valid_sample

    def index2sample(self,index):
        source_utt=self.total_valid_sample['source_utt'][index]
        target_utt=self.total_valid_sample['target_utt'][index]
        source_utt_feat=kaldi_io.read_mat(source_utt)
        target_utt_feat=kaldi_io.read_mat(target_utt)
        return source_utt_feat,target_utt_feat






class parallel_dataset_Genrnal(Dataset):

    def __init__(self, source_utt_scp,target_utt_scp,
                 output_per_step=2):
        #output_per_step解码器每一个step输出的frame数目，输入网络的数据的frame数目必须要是这个的整数倍即seq_length%output_per_step==0
        self.reader=RA_parallel_Reader_General(source_utt_scp,target_utt_scp)

        source_utt_feat,target_utt_feat=self.reader.index2sample(0)
        self.source_utt_feat_dim=source_utt_feat.shape[1]
        self.target_utt_feat_dim=target_utt_feat.shape[1]
        self.output_per_step=output_per_step
        self.round_factor=output_per_step



    def __len__(self):
        return len(self.reader)


    def __getitem__(self,item):
        source_feat,target_feat=self.reader.index2sample(item)
        seq_length=target_feat.shape[0]
        seq_length = _round_up(seq_length, self.output_per_step)
        target_feat = target_feat[0:seq_length, :]
        return source_feat,target_feat


    def collect_fn(self,batch_list): # Dataloader collect_fn
        batch_size=len(batch_list)

        max_length_source_feat=0
        max_length_target_feat=0

        for i in range(batch_size):
            source_feat,target_feat=batch_list[i]

            if max_length_source_feat<source_feat.shape[0]:
                max_length_source_feat=source_feat.shape[0]
            if max_length_target_feat<target_feat.shape[0]:
                max_length_target_feat=target_feat.shape[0]


        source_utt_batch = np.zeros((batch_size,max_length_source_feat,self.source_utt_feat_dim))
        source_utt_length_batch = np.zeros(batch_size, dtype=int)
        target_utt_batch = np.zeros((batch_size,max_length_target_feat,self.target_utt_feat_dim))
        target_utt_length_batch=np.zeros(batch_size,dtype=int)
        for i in range(batch_size):
            source_utt_feat,target_utt_feat = batch_list[i]

            source_utt_batch[i,0:source_utt_feat.shape[0],:]=source_utt_feat
            target_utt_batch[i,0:target_utt_feat.shape[0],:]=target_utt_feat
            source_utt_length_batch[i]=source_utt_feat.shape[0]
            target_utt_length_batch[i]=target_utt_feat.shape[0]

        return source_utt_batch,source_utt_length_batch,target_utt_batch,target_utt_length_batch





if __name__=='__main__':
    from espnet2.VC_SRC.During_training.dataset import infinite_seqlength_optmized_dataloader
    train_dataset = parallel_dataset_Genrnal(
        'path',"",
        output_per_step=2)
    loader=infinite_seqlength_optmized_dataloader(train_dataset,32,num_workers=2,batch_per_group=8)

    t0=time.time()
    for i in range(100):
        BN_feat_batch,mel_target_batch,spk_batch,seq_length_batch=loader.next_batch()
        #time.sleep(2)
        print(BN_feat_batch.shape)
        print(mel_target_batch.shape)
        print(spk_batch)
        print(seq_length_batch)
        print('================')

    print("total time:",time.time()-t0)

