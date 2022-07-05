# -*- coding: utf-8 -*-
import sys
sys.path.append('')
from espnet2.VC_SRC.During_training.dataset_parallel_vc import parallel_dataset_Genrnal
from espnet2.VC_SRC.During_training.load_model import load_model_from_cpd
from torch.utils.data import DataLoader


from espnet2.VC_SRC.evaluation.eval_dataset import eval_dataset
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
train_dataset=parallel_dataset_Genrnal(mynn.source_scp,mynn.target_scp, output_per_step=mynn.hparams.n_frames_per_step)
train_loader=DataLoader(dataset=train_dataset,
                                 batch_size=1,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True,
                                    collate_fn=train_dataset.collect_fn)



eval_dataset=eval_dataset(mynn.source_scp_test)
eval_loader=DataLoader(dataset=eval_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1,
                                  drop_last=False,
                                    collate_fn=eval_dataset.collect_fn)


model=mynn.VC_model(mynn.hparams)

step = load_model_from_cpd(model, checkpoint_dir)
print("inference using step "+str(step))



train_loader_iter=iter(train_loader)
model.eval()
train_total_loss_summary=[]
train_melloss_summary=[]
for i in range(0):
    try:
        source_utt,source_utt_length,target_utt,target_utt_length=train_loader_iter.next()
    except StopIteration:
        train_loader_iter = iter(train_loader)
        source_utt, source_utt_length, target_utt, target_utt_length = train_loader_iter.next()
    source_utt = torch.from_numpy(source_utt).float()
    source_utt_length = torch.from_numpy(source_utt_length)
    target_utt = torch.from_numpy(target_utt).float()
    target_utt_length = torch.from_numpy(target_utt_length)

    with torch.no_grad():
        total_loss, mel_final, state = model(source_utt, source_utt_length, target_utt, target_utt_length)
    print("loss on training set:","total_loss",total_loss)#这里用来确保模型被正确的加载,但是这里正确只能证明参数读取正确，如果inference写错输出结果一样会出问题
    train_total_loss_summary.append(total_loss.item())
    train_melloss_summary.append(state["mel_loss_final"].item())
print("===============================================")
print("average loss on training set:", np.mean(train_total_loss_summary),"mel_loss", np.mean(train_melloss_summary))  # 这里用来确保模型被正确的加载


for uttids,source_utts,source_utts_length in eval_loader:
    source_utts = torch.from_numpy(source_utts).float()
    source_utts_length = torch.from_numpy(source_utts_length)


    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments=model.inference(source_utts,source_utts_length)
    temp=mel_outputs_postnet.cpu().detach().numpy()
    np.save(os.path.join(eval_result_dir, uttids[0]+".npy"),temp[0])








