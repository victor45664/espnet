import sys
sys.path.append('')
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import kaldi_io




class ScpReader(object):  #scp reader 的基础类,read_fn用于从scp路径总读取数据的函数，如果不指定则自适应（可能有性能损失）
    def __init__(self, scp_path, name, read_fn=None,print_detail=True):
        if read_fn is None: #没有手动指定
            self.read_fn = [ np.load,kaldi_io.read_mat]     #可能需要的用于读取的函数
            self.num_of_read_fn = len(self.read_fn)
        elif(callable( read_fn)):
            self._read_data = read_fn
        elif (isinstance(read_fn, list)):
            self.read_fn = read_fn
            self.num_of_read_fn = len(self.read_fn)
        else:
            TypeError("read_fn should be either callable or a list of callable functions")

        self.name=name
        self.scp=pd.read_csv(scp_path,header=None,delim_whitespace=True,names=['key',name])
        if print_detail:
            self.sample_detail()

    def sample_detail(self):
        self.Asample=self.read_data(0)
        if isinstance(self.Asample, np.ndarray):
            print("Found {} {} samples. First {} sample has shape {},mean {},std {}".format(len(self.scp), self.name,
                                                                                        self.scp[ "key" ][ 0 ],
                                                                                        self.Asample.shape,
                                                                                        self.Asample.mean(),
                                                                                        self.Asample.std()))
        else:
            print("Found {} {} samples. First {} sample {}".format(len(self.scp), self.name,
                                                                                        self.scp[ "key" ][ 0 ],
                                                                                        self.Asample))

    def __len__(self):
        return len(self.scp)

    def read_data(self,idx):
        content_path=self.scp[self.name][idx]
        return self.scp[ "key" ][idx],self._read_data(content_path)

    def _read_data(self,content_path):  #从scp的路径中读取数据,这里使用自适应读取，但是效率可能会比较低
        for fni in range(self.num_of_read_fn):    #各个函数的优先级按照list中的顺序决定，如果读取不报错则认为读取成功，尽量还是严格指定读取函数，更加安全
            try:
                data=self.read_fn[fni](content_path)
            except:
                continue
            else: #读取成功了
                if fni!=0:
                    temp=self.read_fn[0]
                    self.read_fn[0] = self.read_fn[fni]
                    self.read_fn[ fni ]=temp #将读取成功的函数移动到最前发，提高效率
                return data
        #走完了for循环，说明全部失败
        raise IOError("Can't read from {}".format(content_path))






def _round_up(length, multiple):
  remainder = length % multiple
  return length if remainder == 0 else length - remainder


class eval_dataset(Dataset):

    def __init__(self, scp_path):
        #output_per_step解码器每一个step输出的frame数目，输入网络的数据的frame数目必须要是这个的整数倍即seq_length%output_per_step==0

        self.reader=ScpReader(scp_path,"utt_feat")

        _,utt_feat=self.reader.read_data(0)
        self.utt_feat_dim=utt_feat.shape[1]
        print("source utt feat dim",self.utt_feat_dim)



    def __len__(self):
        return len(self.reader)


    def __getitem__(self,item):
        key,utt_feat=self.reader.read_data(item)

        utt_feat_length=utt_feat.shape[0]


        return key,utt_feat,utt_feat_length



    def collect_fn(self,batch_list): # Dataloader collect_fn
        batch_size=len(batch_list)

        max_length_in_batch=0
        for i in range(batch_size):
            _,_,utt_feat_length=batch_list[i]
            if max_length_in_batch<utt_feat_length:
                max_length_in_batch=utt_feat_length

        utt_feat_batch = np.zeros((batch_size,max_length_in_batch,self.utt_feat_dim))
        BN_feat_length_batch = np.zeros((batch_size),dtype=int)
        keys=[]
        for i in range(batch_size):
            key,utt_feat,utt_feat_length = batch_list[i]
            utt_feat_batch[i,0:utt_feat_length,:]=utt_feat
            BN_feat_length_batch[i]=utt_feat_length
            keys.append(key)
        return keys,utt_feat_batch,BN_feat_length_batch





class eval_dataset_nonParallel(Dataset):

    def __init__(self, scp_path,output_per_step=2):
        #output_per_step解码器每一个step输出的frame数目，输入网络的数据的frame数目必须要是这个的整数倍即seq_length%output_per_step==0

        self.reader=ScpReader(scp_path,"utt_feat")


        _,utt_feat=self.reader.read_data(0)
        self.utt_feat_dim=utt_feat.shape[1]
        self.output_per_step=output_per_step


    def __len__(self):
        return len(self.reader)


    def __getitem__(self,item):
        key,BN_feat=self.reader.read_data(item)

        BN_feat_seq_length=BN_feat.shape[0]
        BN_feat_seq_length=_round_up(BN_feat_seq_length,self.output_per_step)
        BN_feat= BN_feat[0:BN_feat_seq_length, :]



        return key,BN_feat,BN_feat_seq_length



    def collect_fn(self,batch_list): # Dataloader collect_fn
        batch_size=len(batch_list)

        max_length_in_batch=0
        for i in range(batch_size):
            _,_,BN_feat_seq_length=batch_list[i]
            if max_length_in_batch<BN_feat_seq_length:
                max_length_in_batch=BN_feat_seq_length

        BN_feat_batch = np.zeros((batch_size,max_length_in_batch,self.utt_feat_dim))
        BN_feat_length_batch = np.zeros((batch_size),dtype=int)
        keys=[]
        for i in range(batch_size):
            key,BN_feat,BN_feat_seq_length = batch_list[i]
            BN_feat_batch[i,0:BN_feat_seq_length,:]=BN_feat
            BN_feat_length_batch[i]=BN_feat_seq_length
            keys.append(key)
        return keys,BN_feat_batch,BN_feat_length_batch


if __name__=='__main__':
    scp_path='/home/victor/espnet/egs2/voice_conversion/vc1/data/bdl_mel/bdl_mel.scp'
    raaa=eval_dataset(scp_path)

    for i in range(8):
        key,feat,length=raaa[i]
        print(feat.shape)
        print(length)
        print(key)
        print('================')



