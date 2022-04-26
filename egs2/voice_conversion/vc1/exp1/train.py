# -*- coding: utf-8 -*-
import sys
sys.path.append('')
from espnet2.VC_SRC.During_training.dataset_parallel_vc import parallel_dataset_Genrnal
from espnet2.VC_SRC.During_training.dataset import infinite_seqlength_optmized_dataloader
from espnet2.VC_SRC.VC_utils.gpu_util import get_gpu
import time
import os
from espnet2.VC_SRC.Model_component.VC_utils import change_lr
mutationname=sys.argv[1]
from tensorboardX import SummaryWriter
import torch
mynn = __import__('mutation.' + mutationname, fromlist=True)


modeldir = sys.path[0]

temp = modeldir.split('/')
modelname = temp[-1]

model_save_path = os.path.join(modeldir, 'newest_model_saved', mutationname, mutationname)

loggerdir = modeldir + '/log/' + mutationname


class loss_logger(object):
    def __init__(self,logger_dir):
        self.log_writer=SummaryWriter(log_dir=logger_dir, comment='', purge_step=None, max_queue=10, flush_secs=20, filename_suffix='')
    def add_log(self,step,loss,lr=0):
        self.log_writer.add_scalar('Lr', lr, step)
        self.log_writer.add_scalar('Loss', loss, step)

logger=loss_logger(loggerdir)


train_dataset=parallel_dataset_Genrnal(os.path.join(mynn.datadir,'train'), output_per_step=mynn.hparams.n_frames_per_step)
train_loader=infinite_seqlength_optmized_dataloader(train_dataset,mynn.hparams.batchsize,'train_dataset',num_workers=4,batch_per_group=8)

test_dataset=parallel_dataset_Genrnal(os.path.join(mynn.datadir,'dev'), output_per_step=mynn.hparams.n_frames_per_step)
test_loader=infinite_seqlength_optmized_dataloader(train_dataset,mynn.hparams.batchsize,num_workers=2,batch_per_group=4)


mynn.hparams.BN_feat_dim=train_dataset.BN_feat_dim
mynn.hparams.n_mel_channels=train_dataset.mel_target_dim
mynn.hparams.num_of_spk=train_dataset.num_of_speaker

model=mynn.VC_model(mynn.hparams)


gpu_list=get_gpu(1)
if len(gpu_list)==0:
    sys.exit(1)

model=model.cuda()

optimizer=mynn.optimizer(model.parameters(), lr=mynn.hparams.learning_rate,
                                 weight_decay=mynn.hparams.weight_decay)




print("{},{}:{}, begin".format(modelname, mutationname, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
last_time=time.time()
for step in range(1,mynn.hparams.total_iteration+1):
    lr=mynn.getlr(step)
    change_lr(optimizer,lr)
    BN_feat_train, mel_target_train, spk_train, seq_length_train = train_loader.next_batch()


    if(step%100==0):#记录tensorboard数据，并且保存模型
        print("{},{}:{},delta time={}S, step{}".format(modelname,mutationname,time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),int(time.time()-last_time),step))
        last_time = time.time()
        BN_feat_test,mel_target_test,spk_test,seq_length_test=test_loader.next_batch()
        torch.save(model.state_dict(), model_save_path)

        BN_feat_test=torch.from_numpy(BN_feat_test).float().cuda()
        mel_target_test=torch.from_numpy(mel_target_test).float().cuda()
        seq_length_test=torch.from_numpy(seq_length_test).float().cuda()
        spk_test=torch.from_numpy(spk_test).cuda()

        with torch.no_grad():
            model.eval()
            mel_loss,mel_loss_final,mel_final=model(BN_feat_test,seq_length_test,spk_test,mel_target_test)
            model.train()
        logger.add_log(step,mel_loss_final.item(),lr)
    else:
        BN_feat_train = torch.from_numpy(BN_feat_train).float().cuda()
        mel_target_train = torch.from_numpy(mel_target_train).float().cuda()
        seq_length_train = torch.from_numpy(seq_length_train).float().cuda()
        spk_train = torch.from_numpy(spk_train).cuda()

        mel_loss,mel_loss_final,_ = model(BN_feat_train, seq_length_train, spk_train, mel_target_train)
        (mel_loss+mel_loss_final).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), mynn.hparams.grad_clip_thresh)
        optimizer.step()
        model.zero_grad()














