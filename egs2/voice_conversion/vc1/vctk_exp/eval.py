# -*- coding: utf-8 -*-
import sys
sys.path.append('')
from espnet2.VC_SRC.During_training.dataset_parallel_vc import parallel_dataset_Genrnal
from espnet2.VC_SRC.During_training.dataset import BNfeats_meltarget_spk_dataset,infinite_dataloader,infinite_seqlength_optmized_dataloader

from espnet2.VC_SRC.During_training.load_model import load_model_from_cpd
from torch.utils.data import DataLoader


from espnet2.VC_SRC.evaluation.eval_dataset import eval_dataset_nonParallel as eval_dataset
import os

import numpy as np

mutationname=sys.argv[1]

mynn = __import__('mutation.' + mutationname, fromlist=True)
#import models.multi_spk_deploy_pytorch.mutation.baseline_noatt as mynn
modeldir = sys.path[0]

temp = modeldir.split('/')
modelname = temp[-1]

checkpoint_dir=modeldir + '/newest_model_saved/' + mutationname
eval_result_dir=os.path.join(modeldir , 'eval_results' ,mutationname)

import torch
import random
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
MAX_INF_PER_SPK=4
train_dataset=BNfeats_meltarget_spk_dataset(os.path.join(mynn.datadir,'train'), output_per_step=mynn.hparams.n_frames_per_step)
train_loader=DataLoader(dataset=train_dataset,
                                 batch_size=32,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True,
                                    collate_fn=train_dataset.collect_fn)



eval_dataset=eval_dataset(os.path.join(mynn.datadir,'eval',"ppg_phonetone","phone_post.scp"))
eval_loader=DataLoader(dataset=eval_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1,
                                  drop_last=False,
                                    collate_fn=eval_dataset.collect_fn)

mynn.hparams.BN_feat_dim=train_dataset.BN_feat_dim
mynn.hparams.n_mel_channels=train_dataset.mel_target_dim
mynn.hparams.num_of_spk=train_dataset.num_of_speaker

model=mynn.VC_model(mynn.hparams)

step = load_model_from_cpd(model, checkpoint_dir)
print("inference using step "+str(step))



train_loader_iter=iter(train_loader)
model.eval()
train_total_loss_summary=[]
train_melloss_summary=[]
for i in range(3):
    try:
        BN_feat_train,mel_target_train,spk_train,BN_feat_seq_length=train_loader_iter.next()
    except StopIteration:
        train_loader_iter = iter(train_loader)
        source_utt, source_utt_length, target_utt, target_utt_length = train_loader_iter.next()
    BN_feat_train = torch.from_numpy(BN_feat_train).float()
    mel_target_train = torch.from_numpy(mel_target_train).float()
    seq_length_train = torch.from_numpy(BN_feat_seq_length).float()
    spk_train = torch.from_numpy(spk_train)

    with torch.no_grad():
        total_loss, _, state = model(BN_feat_train, seq_length_train, spk_train, mel_target_train)
    print("loss on training set:","total_loss",total_loss)#这里用来确保模型被正确的加载,但是这里正确只能证明参数读取正确，如果inference写错输出结果一样会出问题
    train_total_loss_summary.append(total_loss.item())
    train_melloss_summary.append(state["mel_loss_final"].item())
print("===============================================")
print("average loss on training set:", np.mean(train_total_loss_summary),"mel_loss", np.mean(train_melloss_summary))  # 这里用来确保模型被正确的加载


os.makedirs(eval_result_dir,exist_ok=True)

count=0
with torch.no_grad():
    for uttids,BN_feat_test,BN_feat_seq_length in eval_loader:
        BN_feat_test = torch.from_numpy(BN_feat_test).float()
        seq_length_test = torch.from_numpy(BN_feat_seq_length).float()
        for spk in range(train_dataset.num_of_speaker):
            spk_test=np.ones((BN_feat_test.shape[0]),dtype=int)*spk


            spk_test=torch.from_numpy(spk_test)


            mel_outputs_postnet,_=model.inference(BN_feat_test,seq_length_test,spk_test)
            temp=mel_outputs_postnet.cpu().detach().numpy()

            os.makedirs(os.path.join(eval_result_dir,str(spk)),exist_ok=True)
            np.save(os.path.join(eval_result_dir,str(spk), uttids[0]+".npy"),temp[0])
        if count>=MAX_INF_PER_SPK:
            break
        count+=1




