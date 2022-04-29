# -*- coding: utf-8 -*-
import sys
sys.path.append('')
from espnet2.VC_SRC.During_training.dataset_parallel_vc import parallel_dataset_Genrnal,infinite_dataloader,infinite_seqlength_optmized_dataloader
from espnet2.VC_SRC.During_training.load_model import load_model_from_cpd
import time
import os
from espnet2.VC_SRC.Model_component.VC_utils import change_lr
mutationname=sys.argv[1]
resum=int(sys.argv[2])
from tensorboardX import SummaryWriter
import torch
mynn = __import__('mutation.' + mutationname, fromlist=True)


modeldir = sys.path[0]

temp = modeldir.split('/')
modelname = temp[-1]


checkpoint_dir=modeldir + '/newest_model_saved/' + mutationname
loggerdir = modeldir + '/log/' + mutationname


class loss_logger(object):
    def __init__(self,logger_dir):
        self.log_writer=SummaryWriter(log_dir=logger_dir, comment='', purge_step=None, max_queue=10, flush_secs=20, filename_suffix='')
    def add_log(self,step,loss,state,lr=0):
        self.log_writer.add_scalar('Lr', lr, step)
        self.log_writer.add_scalar('Loss', loss, step)
        for key in state.keys():
            self.log_writer.add_scalar(key, float(state[key]), step)

logger=loss_logger(loggerdir)


train_dataset=parallel_dataset_Genrnal(mynn.source_scp,mynn.target_scp, output_per_step=mynn.hparams.n_frames_per_step,cache_data=True)
train_loader=infinite_seqlength_optmized_dataloader(train_dataset,mynn.hparams.batchsize,'train_dataset',num_workers=4,max_batchsize_mul_max_length=32*400)

test_dataset=parallel_dataset_Genrnal(mynn.source_scp_test,mynn.target_scp_test, output_per_step=mynn.hparams.n_frames_per_step,cache_data=True)
test_loader=infinite_seqlength_optmized_dataloader(train_dataset,mynn.hparams.batchsize,num_workers=2,max_batchsize_mul_max_length=32*400)


model=mynn.VC_model(mynn.hparams)

start_step=1

if resum:
    start_step = load_model_from_cpd(model, checkpoint_dir)


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model=model.to(device) #vdebug
#model=model #vdebug

optimizer=mynn.optimizer(model.parameters(), lr=mynn.hparams.learning_rate,
                                 weight_decay=mynn.hparams.weight_decay)




print("{},{}:{}, begin".format(modelname, mutationname, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
last_time=time.time()
for step in range(start_step,mynn.hparams.total_iteration+1):
    lr=mynn.getlr(step)
    change_lr(optimizer,lr)
    source_utt,source_utt_length,target_utt,target_utt_length = train_loader.next_batch()


    if(step%100==0):#记录tensorboard数据，并且保存模型
        print("{},{}:{},delta time={}S, step{}".format(modelname,mutationname,time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),int(time.time()-last_time),step))
        last_time = time.time()
        source_utt_test,source_utt_length_test,target_utt_test,target_utt_length_test=test_loader.next_batch()
        if(step%10000==0):
            print("saving model at {}".format(str(step)))
            torch.save(model.state_dict(),
                   modeldir + '/newest_model_saved/{}/s{}_'.format(mutationname, step) + mutationname + '.pth')


        source_utt_test=torch.from_numpy(source_utt_test).float().to(device) #vdebug
        target_utt_test=torch.from_numpy(target_utt_test).float().to(device) #vdebug
        target_utt_length_test=torch.from_numpy(target_utt_length_test).to(device) #vdebug
        source_utt_length_test=torch.from_numpy(source_utt_length_test).to(device) #vdebug



        with torch.no_grad():
            model.eval()
            total_loss,mel_final,state=model(source_utt_test,source_utt_length_test,target_utt_test,target_utt_length_test)
            model.train()
        logger.add_log(step,total_loss.item(),state,lr)
    else:
        source_utt=torch.from_numpy(source_utt).float().to(device) #vdebug
        target_utt=torch.from_numpy(target_utt).float().to(device) #vdebug
        target_utt_length=torch.from_numpy(target_utt_length).to(device) #vdebug
        source_utt_length=torch.from_numpy(source_utt_length).to(device) #vdebug

        loss,_,_ = model(source_utt,source_utt_length,target_utt,target_utt_length)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), mynn.hparams.grad_clip_thresh)
        optimizer.step()
        model.zero_grad()














