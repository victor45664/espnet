# -*- coding: utf-8 -*-
import sys
sys.path.append('')
from espnet2.VC_SRC.During_training.dataset import BNfeats_meltarget_spk_dataset,infinite_dataloader,infinite_seqlength_optmized_dataloader
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
    def add_log(self,step,loss,state,prefix="",lr=0):
        self.log_writer.add_scalar('Lr', lr, step)
        self.log_writer.add_scalar('Loss', loss, step)
        for key in state.keys():
            if key!="hist":
                self.log_writer.add_scalar(prefix+key, float(state[key]), step)
            else:
                for key in state["hist"].keys():
                    self.log_writer.add_histogram(prefix+key, state["hist"][key], step)

logger=loss_logger(loggerdir)


train_dataset=BNfeats_meltarget_spk_dataset(os.path.join(mynn.datadir,'train'), output_per_step=mynn.hparams.n_frames_per_step)
train_loader=infinite_seqlength_optmized_dataloader(train_dataset,mynn.hparams.batchsize,'train_dataset',num_workers=6, batch_per_group=16,max_batchsize_mul_max_length=32*2000)

test_dataset=BNfeats_meltarget_spk_dataset(os.path.join(mynn.datadir,'dev'), output_per_step=mynn.hparams.n_frames_per_step)
test_loader=infinite_seqlength_optmized_dataloader(test_dataset,mynn.hparams.batchsize, batch_per_group=16,num_workers=2,max_batchsize_mul_max_length=32*2000)

mynn.hparams.BN_feat_dim=train_dataset.BN_feat_dim
mynn.hparams.n_mel_channels=train_dataset.mel_target_dim
mynn.hparams.num_of_spk=train_dataset.num_of_speaker

model=mynn.VC_model(mynn.hparams)

start_step=1
print("begin")
if resum:
    start_step = load_model_from_cpd(model, checkpoint_dir)


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model=model.to(device) #vdebug


optimizer=mynn.optimizer(model.parameters(), lr=mynn.hparams.learning_rate,
                                 weight_decay=mynn.hparams.weight_decay)




print("{},{}:{}, begin".format(modelname, mutationname, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
last_time=time.time()
for step in range(start_step,mynn.hparams.total_iteration+1):
    lr=mynn.getlr(step)
    change_lr(optimizer,lr)
    BN_feat_train, mel_target_train, spk_train, seq_length_train = train_loader.next_batch()
    
    BN_feat_train = torch.from_numpy(BN_feat_train).float().to(device)
    mel_target_train = torch.from_numpy(mel_target_train).float().to(device)
    seq_length_train = torch.from_numpy(seq_length_train).float().to(device)
    spk_train = torch.from_numpy(spk_train).to(device)


    loss, _, train_state = model(BN_feat_train, seq_length_train, spk_train, mel_target_train)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), mynn.hparams.grad_clip_thresh)
    optimizer.step()
    model.zero_grad()
    if(step%100==0):#记录tensorboard数据，并且保存模型
        print("{},{}:{},delta time={}S, step{}".format(modelname,mutationname,time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),int(time.time()-last_time),step))
        last_time = time.time()
        BN_feat_test,mel_target_test,spk_test,seq_length_test=test_loader.next_batch()
        BN_feat_test=torch.from_numpy(BN_feat_test).float().to(device)
        mel_target_test=torch.from_numpy(mel_target_test).float().to(device)
        seq_length_test=torch.from_numpy(seq_length_test).float().to(device)
        spk_test=torch.from_numpy(spk_test).to(device)
        
        with torch.no_grad():
            model.eval()
            total_loss,mel_final,state = model(BN_feat_test, seq_length_test, spk_test, mel_target_test)
            model.train()
      

        logger.add_log(step,total_loss.item(),state,prefix="test_",lr=lr)
        logger.add_log(step, loss.item(), train_state, prefix="train_", lr=lr)
        if(step%10000==0):
            print("saving model at {}".format(str(step)))
            torch.save(model.state_dict(),
                   modeldir + '/newest_model_saved/{}/s{}_'.format(mutationname, step) + mutationname + '.pth')
















