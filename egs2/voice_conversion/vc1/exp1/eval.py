# -*- coding: utf-8 -*-
import sys
sys.path.append('')
from espnet2.VC_SRC.During_training import BNfeats_meltarget_spk_dataset
from torch.utils.data import DataLoader
from espnet2.VC_SRC.VC_utils import get_gpu
from espnet2.VC_SRC import result_to_dir
from espnet2.VC_SRC.evaluation.eval_dataset import eval_dataset
import os

import numpy as np
from espnet2.VC_SRC import gen_tx_hifigan as vocoder
mutationname=sys.argv[1]

mynn = __import__('mutation.' + mutationname, fromlist=True)
#import models.multi_spk_deploy_pytorch.mutation.baseline_noatt as mynn
modeldir = sys.path[0]

temp = modeldir.split('/')
modelname = temp[-1]

model_save_path = os.path.join(modeldir , 'newest_model_saved' ,mutationname,mutationname)
eval_result_dir=os.path.join(modeldir , 'eval_result' ,mutationname)

import torch
import random
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
train_dataset=BNfeats_meltarget_spk_dataset(os.path.join(mynn.datadir,'train'), output_per_step=mynn.hparams.n_frames_per_step)
train_loader=DataLoader(dataset=train_dataset,
                                  batch_size=mynn.hparams.batchsize,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True,
                                    collate_fn=train_dataset.collect_fn)

batchsize=8

eval_dataset=eval_dataset(os.path.join(mynn.datadir,'eval'), output_per_step=mynn.hparams.n_frames_per_step)
eval_loader=DataLoader(dataset=eval_dataset,
                                  batch_size=batchsize,
                                  shuffle=False,
                                  num_workers=1,
                                  drop_last=False,
                                    collate_fn=eval_dataset.collect_fn)

mynn.hparams.BN_feat_dim=train_dataset.BN_feat_dim
mynn.hparams.n_mel_channels=train_dataset.mel_target_dim
mynn.hparams.num_of_spk=train_dataset.num_of_speaker

model=mynn.VC_model(mynn.hparams)


gpu_list=get_gpu(1)
if len(gpu_list)==0:
    sys.exit(1)

model.load_state_dict(torch.load(model_save_path))
model.cuda()


train_loader_iter=iter(train_loader)
model.eval()
train_l21loss_summary=[]
for i in range(10):
    try:
        BN_feat_train,mel_target_train,spk_train,BN_feat_seq_length=train_loader_iter.next()
    except StopIteration:
        train_loader_iter = iter(train_loader)

    BN_feat_train = torch.from_numpy(BN_feat_train).float().cuda()
    mel_target_train = torch.from_numpy(mel_target_train).float().cuda()
    seq_length_train = torch.from_numpy(BN_feat_seq_length).float().cuda()
    spk_train = torch.from_numpy(spk_train).cuda()
    with torch.no_grad():
        mel_loss, mel_loss_final, _ = model(BN_feat_train, seq_length_train, spk_train, mel_target_train)
    print("loss on training set:","total_loss",mel_loss_final)#这里用来确保模型被正确的加载,但是这里正确只能证明参数读取正确，如果inference写错输出结果一样会出问题
    train_l21loss_summary.append(mel_loss_final.item())

print("===============================================")
print("average loss on training set:", "mel_loss", np.mean(train_l21loss_summary))  # 这里用来确保模型被正确的加载


for uttids,BN_feat_test,BN_feat_seq_length in eval_loader:
    BN_feat_test = torch.from_numpy(BN_feat_test).float().cuda()
    seq_length_test = torch.from_numpy(BN_feat_seq_length).float().cuda()
    for spk in range(train_dataset.num_of_speaker):
        spk_test=np.ones((BN_feat_test.shape[0]),dtype=int)*spk


        spk_test=torch.from_numpy(spk_test).cuda()

        with torch.no_grad():
            mel_outputs_postnet,residural_mse=model.inference(BN_feat_test,seq_length_test,spk_test)
        temp=mel_outputs_postnet.cpu().detach().numpy()
        result_to_dir(uttids,temp,BN_feat_seq_length,os.path.join(eval_result_dir,str(spk)),vocoder=vocoder)
        print("done eval spk:",str(spk))
    break






