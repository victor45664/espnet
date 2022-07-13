# -*- coding: utf-8 -*-
import sys
sys.path.append('')
from torch.utils.data import Dataset,DataLoader
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






class RA_ali_BNfeats_Reader_General(object):
    def __init__(self,BNfeats_scp_file_path,ali_scp_file_path):
        ali=pd.read_csv(ali_scp_file_path,header=None,delim_whitespace=True,names=['key','alignment'])
        BNfeats=pd.read_csv(BNfeats_scp_file_path,header=None,delim_whitespace=True,names=['key','BNfeat'])
        print("find {} BNfeat".format(len(BNfeats)))
        print("find {} alignment".format(len(ali)))
        total_training_sample=pd.merge(ali,BNfeats,how='outer',on='key')
        self.total_valid_sample=total_training_sample.dropna(axis=0, how='any')  #valid sample 是指两个都有数据的样本
        self.total_valid_sample=self.total_valid_sample.reset_index(drop=True)
        self.num_of_valid_sample=len(self.total_valid_sample)
        print("{} valid sample".format(self.num_of_valid_sample))


    def __len__(self):
        return self.num_of_valid_sample

    def index2sample(self,index):
        ali_path=self.total_valid_sample['alignment'][index]
        feat_path=self.total_valid_sample['BNfeat'][index]
        feat=kaldi_io.read_mat(feat_path)
        ali=kaldi_io.read_vec_int(ali_path)
        return feat,ali





class BNfeat_ali_dataset(Dataset):
    #这个dataset输出BN特征和对于的ali，同时输出随机的target speaker，这是为了之后的多人VC做准备的，
    def __init__(self, data_dir,number_of_spk=1,output_per_step=2, max_length_allowed=300):
        #output_per_step解码器每一个step输出的frame数目，输入网络的数据的frame数目必须要是这个的整数倍即seq_length%output_per_step==0

        self.reader=RA_ali_BNfeats_Reader_General(
            os.path.join(data_dir,"ppg_phonetone","phone_post.scp"),
            os.path.join(data_dir,"ali.scp")
        )

        BN_feat,ali=self.reader.index2sample(0)
        self.BN_feat_dim=BN_feat.shape[1]
        print("BN feat dim",self.BN_feat_dim)
        self.number_of_spk=number_of_spk
        self.output_per_step=output_per_step
        self.max_length_allowed=_round_up(max_length_allowed,self.output_per_step)


    def __len__(self):
        return len(self.reader)


    def __getitem__(self,item):

        BN_feat,ali=self.reader.index2sample(item)   #BN_feat,ali的长度一定一样长，重点是VC模型输出的要和ali长度一致
        seq_length=BN_feat.shape[0]

        if seq_length>self.max_length_allowed:
            s0=random.randint(0, seq_length - self.max_length_allowed)
            BN_feat= BN_feat[s0:s0+self.max_length_allowed, :]
            seq_length=self.max_length_allowed
        else:
            seq_length=_round_up(seq_length,self.output_per_step)
            BN_feat= BN_feat[0:seq_length, :]


        ali= ali[0:seq_length]
        speaker=random.randint(0,self.number_of_spk-1)  #随机选择一个target speaker
        return BN_feat,ali,speaker,seq_length



    def collect_fn(self,batch_list): # Dataloader collect_fn
        batch_size=len(batch_list)

        max_length_in_batch=0

        for i in range(batch_size):
            _,_,_,seq_length=batch_list[i]

            if max_length_in_batch<seq_length:
                max_length_in_batch=seq_length



        BN_feat_batch = np.zeros((batch_size,max_length_in_batch,self.BN_feat_dim))
        ali_batch = np.zeros((batch_size,max_length_in_batch))
        spk_batch = np.zeros((batch_size),dtype=int)
        seq_length_batch=np.zeros(batch_size,dtype=int)
        for i in range(batch_size):
            BN_feat,ali,spk,seq_length = batch_list[i]
            BN_feat_batch[i,0:seq_length,:]=BN_feat
            ali_batch[i,0:seq_length]=ali    #ali_和BN_feat一定等长
            spk_batch[i]=spk
            seq_length_batch[i]=seq_length

        return BN_feat_batch,ali_batch,spk_batch,seq_length_batch






class RA_BN_feats_mel_target_speaker_Reader_General(object):   #读取
    def __init__(self,data_dir):
        BN_feats_scp_path=os.path.join(data_dir,"ppg_phonetone","phone_post.scp")     #BN
        mel_target_scp_path=os.path.join(data_dir,"mel_target.scp")
        speaker_path=os.path.join(data_dir,"utt2spk_id")

        BN_feats_scp=pd.read_csv(BN_feats_scp_path,header=None,delim_whitespace=True,names=['key','BN_feat'])
        mel_target_scp=pd.read_csv(mel_target_scp_path,header=None,delim_whitespace=True,names=['key','mel_target'])
        speaker=pd.read_csv(speaker_path,header=None,delim_whitespace=True,names=['key','spk'])
        self.num_of_speaker=speaker['spk'].max()+1   #spk的编号是从0----num_of_spk-1


        print("find {} utts with BN_feat".format(len(BN_feats_scp)))
        print("find {} utts with mel_target".format(len(mel_target_scp)))
        print("find {} utts with speaker info".format(len(speaker)))    #如果数据准备没有问题，这三个应该相等，
                                                                        # 但如果不相等也没问题，后面会除无效的训练样本


        key_BN_mel=pd.merge(BN_feats_scp,mel_target_scp,how='outer',on='key')
        total_training_sample=pd.merge(key_BN_mel,speaker,how='outer',on='key')

        self.total_training_sample=total_training_sample.dropna(axis=0,how='any')
        self.total_training_sample=self.total_training_sample.reset_index(drop=True)
        self.num_of_training_sample=len(self.total_training_sample)
        print("using {} utt and {} speakers".format(self.num_of_training_sample,self.num_of_speaker))

    def save_total_training_sample(self,save_path):
        self.total_training_sample.to_csv(save_path)




    def __len__(self):
        return self.num_of_training_sample

    def index2sample(self,index):
        BN_feat_path=self.total_training_sample['BN_feat'][index]
        mel_target_path=self.total_training_sample['mel_target'][index]
        speaker = int(self.total_training_sample['spk'][index])

        BN_feat=kaldi_io.read_mat(BN_feat_path)
        mel_target=np.load(mel_target_path)

        return BN_feat,mel_target,speaker



    def key2sample(self,key):
        pass





class BNfeats_meltarget_spk_dataset_Genrnal(Dataset):   #这个dataset就是用来训练普通的VC模型的，返回BN特征和meltarget以及speaker
                                                        # ,这个datase不t处理meltarget和BN特征不等长的问题，因此时间降采样后meltarget和BN特征长度会不一致，即使没有时间降采样，两者也会有一到两帧的差别
                                                        #这会导致计算MSEloss的时候出错，因此需要在网络中特别处理这个问题

    def __init__(self, data_dir,output_per_step=2,time_subsample_rate=4,max_length_allowed=300):
        #output_per_step解码器每一个step输出的frame数目，输入网络的数据的frame数目必须要是这个的整数倍即seq_length%output_per_step==0
        #time_subsample_rate BN特征被降采样的倍数，在输入网络后这些BN特征会被升采样对应的倍数，因此可以认为BN特征的长度是time_subsample_rate倍
        self.reader=RA_BN_feats_mel_target_speaker_Reader_General(data_dir)

        BN_feat,mel_target,speaker=self.reader.index2sample(0)
        self.BN_feat_dim=BN_feat.shape[1]
        self.mel_target_dim=mel_target.shape[1]
        self.num_of_speaker=self.reader.num_of_speaker
        self.output_per_step=output_per_step
        self.time_subsample_rate=time_subsample_rate#BN特征被降采样的倍数
        self.round_factor=lcm(self.output_per_step, self.time_subsample_rate) #最小公倍数，长度必须要被这个整除
        self.max_length_allowed=_round_up(max_length_allowed, self.round_factor) #确保max——length能被这两个整除,下面的所有除法都要确保得到整数



    def __len__(self):
        return len(self.reader)


    def __getitem__(self,item):

        BN_feat,mel_target,speaker=self.reader.index2sample(item)
        BN_feat_seq_length=BN_feat.shape[0]

        if BN_feat_seq_length*self.time_subsample_rate > self.max_length_allowed:
            s0=random.randint(0, BN_feat_seq_length - int(self.max_length_allowed/self.time_subsample_rate))
            BN_feat= BN_feat[s0:int(s0+self.max_length_allowed/self.time_subsample_rate), :]
            BN_feat_seq_length=int(self.max_length_allowed/self.time_subsample_rate)
        else:

            BN_feat_seq_length=int(_round_up(BN_feat_seq_length*self.time_subsample_rate,self.round_factor)/self.time_subsample_rate)

            BN_feat= BN_feat[0:BN_feat_seq_length, :]

        mel_target=change_length(mel_target,BN_feat.shape[0]*self.time_subsample_rate)

        return BN_feat,mel_target,speaker,BN_feat_seq_length,mel_target.shape[0]


    def collect_fn(self,batch_list): # Dataloader collect_fn
        batch_size=len(batch_list)

        max_length_BN_feat=0

        for i in range(batch_size):
            _,_,_,BN_feat_seq_length,mel_target_seq_length=batch_list[i]

            if max_length_BN_feat<BN_feat_seq_length:
                max_length_BN_feat=BN_feat_seq_length



        BN_feat_batch = np.zeros((batch_size,max_length_BN_feat,self.BN_feat_dim))
        mel_target_batch = np.zeros((batch_size,max_length_BN_feat*self.time_subsample_rate,self.mel_target_dim))
        spk_batch = np.zeros((batch_size),dtype=int)
        seq_length_batch=np.zeros(batch_size,dtype=int)
        for i in range(batch_size):
            BN_feat,mel_target,spk,BN_feat_seq_length,mel_target_seq_length = batch_list[i]
            BN_feat_batch[i,0:BN_feat_seq_length,:]=BN_feat
            mel_target_batch[i,0:mel_target_seq_length,:]=mel_target    #mel_target_和BN_feat不一定等长，mel_target一般会长一点
            spk_batch[i]=spk
            seq_length_batch[i]=mel_target_seq_length

        return BN_feat_batch,mel_target_batch,spk_batch,seq_length_batch   #这里的seq_length是meltarget的seq_length





class BNfeats_meltarget_spk_dataset(Dataset):   #这个dataset就是用来训练普通的VC模型的，返回BN特征和meltarget以及speaker

    def __init__(self, data_dir,output_per_step=2, max_length_allowed=3000):
        #output_per_step解码器每一个step输出的frame数目，输入网络的数据的frame数目必须要是这个的整数倍即seq_length%output_per_step==0

        self.reader=RA_BN_feats_mel_target_speaker_Reader_General(data_dir)

        BN_feat,mel_target,speaker=self.reader.index2sample(0)
        self.BN_feat_dim=BN_feat.shape[1]
        self.mel_target_dim=mel_target.shape[1]
        self.num_of_speaker=self.reader.num_of_speaker

        self.output_per_step=output_per_step
        self.max_length_allowed=_round_up(max_length_allowed,self.output_per_step)
        print("BN feat dim is {}".format(self.BN_feat_dim))

    def __len__(self):
        return len(self.reader)


    def __getitem__(self,item):

        BN_feat,mel_target,speaker=self.reader.index2sample(item)
        BN_feat_seq_length=BN_feat.shape[0]

        if BN_feat_seq_length>self.max_length_allowed:
            s0=random.randint(0, BN_feat_seq_length - self.max_length_allowed)
            BN_feat= BN_feat[s0:s0+self.max_length_allowed, :]
            BN_feat_seq_length=self.max_length_allowed
        else:
            BN_feat_seq_length=_round_up(BN_feat_seq_length,self.output_per_step)
            BN_feat= BN_feat[0:BN_feat_seq_length, :]



        mel_target_seq_length = mel_target.shape[0]
        if mel_target_seq_length > self.max_length_allowed:
            s0=random.randint(0, mel_target_seq_length - self.max_length_allowed)
            mel_target= mel_target[s0:s0+self.max_length_allowed]
            mel_target_seq_length=self.max_length_allowed
        else:
            mel_target_seq_length=_round_up(mel_target_seq_length,self.output_per_step)
            mel_target= mel_target[0:mel_target_seq_length, :]

        return BN_feat,mel_target,speaker,BN_feat_seq_length,mel_target_seq_length



    def collect_fn(self,batch_list): # Dataloader collect_fn
        batch_size=len(batch_list)

        max_length_in_batch=0

        for i in range(batch_size):
            _,_,_,BN_feat_seq_length,mel_target_seq_length=batch_list[i]

            if max_length_in_batch<BN_feat_seq_length:
                max_length_in_batch=BN_feat_seq_length
            if max_length_in_batch<mel_target_seq_length:
                max_length_in_batch=mel_target_seq_length


        BN_feat_batch = np.zeros((batch_size,max_length_in_batch,self.BN_feat_dim))
        mel_target_batch = np.zeros((batch_size,max_length_in_batch,self.mel_target_dim))
        spk_batch = np.zeros((batch_size),dtype=int)
        seq_length_batch=np.zeros(batch_size,dtype=int)
        for i in range(batch_size):
            BN_feat,mel_target,spk,BN_feat_seq_length,mel_target_seq_length = batch_list[i]
            BN_feat_batch[i,0:BN_feat_seq_length,:]=BN_feat
            mel_target_batch[i,0:mel_target_seq_length,:]=mel_target    #mel_target_和BN_feat不一定等长，mel_target一般会长一点
            spk_batch[i]=spk
            seq_length_batch[i]=mel_target_seq_length

        return BN_feat_batch,mel_target_batch,spk_batch,seq_length_batch

    def collect_fn_diff_length(self,batch_list): # Dataloader collect_fn
        batch_size=len(batch_list)
        max_BN_feat_length_in_batch=0
        max_mel_target_length_in_batch=0

        for i in range(batch_size):
            _,_,_,BN_feat_seq_length,mel_target_seq_length=batch_list[i]
            if max_BN_feat_length_in_batch<BN_feat_seq_length:
                max_BN_feat_length_in_batch=BN_feat_seq_length
            if max_mel_target_length_in_batch<mel_target_seq_length:
                max_mel_target_length_in_batch=mel_target_seq_length


        BN_feat_batch = np.zeros((batch_size,max_BN_feat_length_in_batch,self.BN_feat_dim))
        mel_target_batch = np.zeros((batch_size,max_mel_target_length_in_batch,self.mel_target_dim))
        spk_batch = np.zeros((batch_size),dtype=int)

        for i in range(batch_size):
            BN_feat,mel_target,spk,BN_feat_seq_length,mel_target_seq_length = batch_list[i]
            BN_feat_batch[i,0:BN_feat_seq_length,:]=BN_feat
            mel_target_batch[i,0:mel_target_seq_length,:]=mel_target    #mel_target_和BN_feat不一定等长，mel_target一般会长一点
            spk_batch[i]=spk



        return BN_feat_batch,mel_target_batch,spk_batch  #




class infinite_dataloader(object):
    def __init__(self,dataset,batchsize,num_workers,log_string):
        self.log_string=log_string
        self.dataloader=DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  collate_fn=dataset.collect_fn)
        self.dataloader_iter=iter(self.dataloader)
        self.epoch=0

    def next_batch(self):
        try:
            self.output = self.dataloader_iter.next()
        except StopIteration:
            self.epoch = self.epoch + 1
            print("{},{} epoch{}".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), self.log_string,
                                         self.epoch))

            self.dataloader_iter = iter(self.dataloader)   #如果一个epoch结束就把上次的数据再返回一次，可以节省时间



        return self.output







#
#
#
# class seqlength_optmized_dataset(Dataset):   #这个dataset就是用来训练普通的VC模型的，返回BN特征和meltarget以及speaker
#
#     def __init__(self,dataloader,batch_per_group,batchsize):   #直接返回一个group，如果group长度不够,就扔掉
#         self.dataloader=dataloader
#         self.dataloader_iter=iter(dataloader)
#         self.batch_per_group=batch_per_group
#         self.batchsize=batchsize
#         self.remaining_group=self.__len__()
#
#
#     def __len__(self):
#         num_of_group=int(len(self.dataloader)/self.batch_per_group/self.batchsize)
#
#
#         return num_of_group
#
#
#
#     def reset_dataloader(self):
#         self.dataloader_iter=iter(self.dataloader)
#         self.remaining_group=self.len()
#
#     def __getitem__(self,index):
#         sample_in_groups=[]
#         for i in range(self.batch_per_group*self.batchsize):
#             sample_in_groups.append(self.dataloader_iter.next())
#
#         print(sample_in_groups[0])
#         sample_in_groups.sort(key=lambda x: x[0].shape[0])  #第一个数据的1维度就是长度，按照长度排序
#
#         self.remaining_group-=1
#
#         return  sample_in_groups   #已经根据length排好序的样本
#
#
#
# class infinite_seqlength_optmized_dataloader(object):  #这个dataloader真对序列不等长进行了优化，把长度相近的序列都聚到了一起
#     def __init__(self,dataset,batchsize,log_string,num_workers=4,batch_per_group=32):
#         self.log_string=log_string
#         self.dataset=dataset
#         self.batch_per_group=batch_per_group
#         single_sample_dataloader=DataLoader(dataset=self.dataset,
#                                                batch_size=1,
#                                                shuffle=True,
#                                                num_workers=num_workers,
#                                                drop_last=True,
#                                                collate_fn=lambda x: x[0])
#         self.batchsize=batchsize
#         self.group_data_dataset=seqlength_optmized_dataset(single_sample_dataloader,
#                                                            batch_per_group,self.batchsize)
#
#         self.loading_group=0
#         self.loading_batch=0
#
#         self.epoch=0
#         self.next_group()
#
#     def next_group(self):
#         self.current_group = self.group_data_dataset[self.loading_group]
#         self.loading_group += 1
#
#         if self.loading_group>len(self.group_data_dataset):
#             self.group_data_dataset.reset_dataloader()
#             self.epoch = self.epoch + 1
#             print("{},{} epoch{}".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), self.log_string,
#                                          self.epoch))  #一个epoch了
#
#             self.loading_group=0
#
#
#
#
#
#     def next_batch(self):
#         if self.loading_batch>self.batch_per_group:
#             self.loading_batch=0
#             self.next_group()
#         self.loading_batch+=1
#
#         current_batch=self.current_group[self.loading_batch*self.batchsize:(self.loading_batch+1)*self.batchsize]
#
#
#
#         return self.dataset.collect_fn(current_batch)




def group_length_sorting_collate_fn(batch_list):
    batch_list.sort(key=lambda x: x[0].shape[0])  # 第一个数据的1维度就是长度，按照长度排序
    return batch_list




import time

class  infinite_seqlength_optmized_dataloader(object):  # 这个dataloader针对序列不等长进行了优化，把长度相近的序列都聚到了一起
                                                        #长度以dataset返回的第一个.shape[0]为准，参考group_length_sorting_collate_fn函数
    def __init__(self, dataset, batchsize, log_string=None, num_workers=4, batch_per_group=32,max_batchsize_mul_max_length=32*2000,min_batch_size=16):
                                                            # max_batchsize_mul_max_length是指batchsize*max_length的最大值，如果超过会自动减小batchsize，这是为了防止现存爆炸,这个不能太小，否则会出错
        self.log_string = log_string                        #min_batch_size是最小允许的batchsize，防止seq_length太长导致batchsize太小
        self.dataset = dataset
        self.batch_per_group = batch_per_group
        self.batchsize = batchsize
        self.max_batchsize_mul_max_length=max_batchsize_mul_max_length
        self.min_batch_size=min(min_batch_size,batchsize-1)
        self.group_dataloader = DataLoader(dataset=self.dataset,
                                              batch_size=self.batch_per_group*self.batchsize,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              drop_last=False,
                                              collate_fn=group_length_sorting_collate_fn)

        self.single_sample_dataloader_iter1=iter(self.group_dataloader)
        self.single_sample_dataloader_iter2=iter(self.group_dataloader)
        self.using_loader=1    #正在使用的loader
        self.epoch = 0
        self.loading_group=0
        self.residual_sample=[] #过长的序列，可能会被抛弃，因此攒起来
        self.next_group()
        self._next_batch_thread = threading.Thread(target=self._next_batch)
        self._next_batch_thread.start()



    def next_group(self):
        if self.using_loader == 1:
            self.current_group = self.single_sample_dataloader_iter1.next()
        else:
            self.current_group = self.single_sample_dataloader_iter2.next()

        self.loading_sample = 0

        self.loading_group += 1

        if self.loading_group >= len(self.group_dataloader):
            if self.using_loader == 1:
                self.single_sample_dataloader_iter1 = iter(self.group_dataloader)
                self.using_loader=2
            else:
                self.single_sample_dataloader_iter2 = iter(self.group_dataloader)
                self.using_loader=1

            self.epoch = self.epoch + 1
            self.loading_group = 0
            if self.log_string!=None:
                print(
                    "{},{} epoch{}".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), self.log_string,
                                           self.epoch))  # 一个epoch了

    def next_batch(self):
        self._next_batch_thread.join()
        cb = self.current_batch
        self._next_batch_thread = threading.Thread(target=self._next_batch)
        self._next_batch_thread.start()
        return cb

    def _next_batch(self):

        if len(self.residual_sample)>self.min_batch_size:
            self.current_batch= self.dataset.collect_fn(self.residual_sample)
            self.residual_sample=[]
            return self.current_batch    #如果收集到了足够的residual sample就使用这些，否则正常next batch




        if self.loading_sample+self.min_batch_size > len(self.current_group):
            self.residual_sample.extend(self.current_group[self.loading_sample:])
            self.next_group()


        sp=self.loading_sample  #起始样品指针
        ep=min(self.loading_sample+self.batchsize,len(self.current_group)-1)  #终止样品指针

        while 1:
            cb=self.current_group[ep]
            if cb[0].shape[0]*(ep-sp)>self.max_batchsize_mul_max_length:
                ep-=1
            else:
                break    #这里长度以第一个的.shape[0] 为准


        current_batch = self.current_group[sp:ep]
        self.loading_sample =ep

        if len(current_batch)<self.min_batch_size:
            self.residual_sample.extend(current_batch)
            self.next_group()
            return self.current_batch  #样本太少了，直接返回上一个batch，导致这个情况有可能是group中剩余样本太少，或者之后的序列都太长，either way we should get next group
        else:
            self.current_batch=self.dataset.collect_fn(current_batch)   #这里是给下一个batch预备的，如果下一个batch数据不足，就直接返回这个
            return self.current_batch


if __name__=='__main__':
    train_dataset = BNfeats_meltarget_spk_dataset(
        'path',
        output_per_step=2, max_length_allowed=4000)
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

