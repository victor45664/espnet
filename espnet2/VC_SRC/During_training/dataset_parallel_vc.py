# -*- coding: utf-8 -*-
import sys
sys.path.append('')
from torch.utils.data import Dataset,DataLoader
import time
import pandas as pd

import threading
import numpy as np


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
    def __init__(self, source_utt_scp_file_path, target_utt_scp_file_path,cache_data=False):
        source_utt_scp=pd.read_csv(target_utt_scp_file_path, header=None, delim_whitespace=True, names=['key', 'source_utt'])
        target_utt_scp=pd.read_csv(source_utt_scp_file_path, header=None, delim_whitespace=True, names=['key', 'target_utt'])
        print("find {} source_utt".format(len(source_utt_scp)))
        print("find {} target_utt".format(len(target_utt_scp)))
        total_training_sample=pd.merge(source_utt_scp,target_utt_scp,how='outer',on='key')
        self.total_valid_sample=total_training_sample.dropna(axis=0, how='any')  #valid sample 是指两个都有数据的样本
        self.total_valid_sample=self.total_valid_sample.reset_index(drop=True)
        self.num_of_valid_sample=len(self.total_valid_sample)
        print("{} valid sample".format(self.num_of_valid_sample))
        self.cache_data=False
        if cache_data:
            self.cache=[]
            for i in range(self.num_of_valid_sample):
                self.cache.append(self.index2sample(i))
            self.cache_data = True

    def __len__(self):
        return self.num_of_valid_sample

    def index2sample(self,index):

        if self.cache_data:
            source_utt_feat,target_utt_feat=self.cache[index]

        else:
            source_utt_path = self.total_valid_sample['source_utt'][index]
            target_utt_path = self.total_valid_sample['target_utt'][index]
            source_utt_feat=np.load(source_utt_path)
            target_utt_feat=np.load(target_utt_path)


        return source_utt_feat,target_utt_feat






class parallel_dataset_Genrnal(Dataset):

    def __init__(self, source_utt_scp,target_utt_scp,
                 output_per_step=2,cache_data=False):
        #output_per_step解码器每一个step输出的frame数目，输入网络的数据的frame数目必须要是这个的整数倍即seq_length%output_per_step==0
        self.reader=RA_parallel_Reader_General(source_utt_scp,target_utt_scp,cache_data=cache_data)

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


def group_length_sorting_collate_fn(batch_list):
    batch_list.sort(key=lambda x: x[0].shape[0])  # 第一个数据的1维度就是长度，按照长度排序
    return batch_list


class   infinite_seqlength_optmized_dataloader(object):  # 这个dataloader针对序列不等长进行了优化，把长度相近的序列都聚到了一起
                                                        #长度以dataset返回的第一个.shape[0]为准，参考group_length_sorting_collate_fn函数
    def __init__(self, dataset, batchsize, log_string=None, num_workers=4, batch_per_group=8,max_batchsize_mul_max_length=48*400,min_batch_size=16):
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

        self.single_sample_dataloader_iter=iter(self.group_dataloader)
        self.epoch = 0
        self.loading_group=0
        self.residual_sample=[] #过长的序列，可能会被抛弃，因此攒起来
        self.next_group()
        self._next_batch_thread = threading.Thread(target=self._next_batch)
        self._next_batch_thread.start()



    def next_group(self):
        self.current_group = self.single_sample_dataloader_iter.next()
        self.loading_sample = 0

        self.loading_group += 1

        if self.loading_group >= len(self.group_dataloader):
            self.single_sample_dataloader_iter = iter(self.group_dataloader)
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



class   infinite_dataloader(object):  # 这个dataloader针对序列不等长进行了优化，把长度相近的序列都聚到了一起
                                                        #长度以dataset返回的第一个.shape[0]为准，参考group_length_sorting_collate_fn函数
    def __init__(self, dataset, batchsize, log_string=None, num_workers=4,):
                                                            # max_batchsize_mul_max_length是指batchsize*max_length的最大值，如果超过会自动减小batchsize，这是为了防止现存爆炸,这个不能太小，否则会出错
        self.log_string = log_string                        #min_batch_size是最小允许的batchsize，防止seq_length太长导致batchsize太小
        self.dataset = dataset
        self.batchsize = batchsize
        self.dataloader = DataLoader(dataset=self.dataset,
                                              batch_size=batchsize,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              drop_last=False,
                                              collate_fn=self.dataset.collect_fn)

        self.dataloader_iter=iter(self.dataloader)
        self.epoch = 0



    def next_batch(self):
        try:
            cb = self.dataloader_iter.next()
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            cb = self.dataloader_iter.next()
            if self.log_string!=None:
                self.epoch+=1
                print(
                    "{},{} epoch{}".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), self.log_string,
                                           self.epoch))  # 一个epoch了
        return cb



if __name__=='__main__':

    train_dataset = parallel_dataset_Genrnal(
        '/home/victor/espnet/egs2/voice_conversion/vc1/data/source.scp','/home/victor/espnet/egs2/voice_conversion/vc1/data/target.scp',
        output_per_step=1)

    loader=infinite_seqlength_optmized_dataloader(train_dataset,2,num_workers=2,batch_per_group=8)

    t0=time.time()
    for i in range(100):
        source_utt,source_utt_length,target_utt,target_utt_length=loader.next_batch()
        #time.sleep(2)
        print(source_utt.shape)
        print(source_utt.mean())
        print(source_utt_length)
        print(target_utt.shape)
        print(target_utt_length)
        print('================')

    print("total time:",time.time()-t0)

