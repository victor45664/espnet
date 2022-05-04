



import os
import torch


def get_step_from_checkpoint(fname):
    return int(fname.split("_")[0][1:])

def load_model_from_cpd(Model,chekpoint_dir):
    if not os.path.exists(chekpoint_dir):
        return 1
    checkpoints=os.listdir(chekpoint_dir)
    newest_step=0
    newest_model_fname=""
    for checkpoint in checkpoints:
        step=get_step_from_checkpoint(checkpoint)
        if step>newest_step:
            newest_step=step
            newest_model_fname=checkpoint
    if newest_model_fname=="":
        return 1
    else:
        print("resuming at step",newest_step)
        newest_model_savepath=os.path.join(chekpoint_dir,newest_model_fname)
        Model.load_state_dict(torch.load(newest_model_savepath, map_location='cpu'))

    return newest_step


if __name__ == '__main__':
    cpd="/home/yufei/HUW4/models/test_A/newest_model_saved/densenet169_nonlocal_dot_amsoftmax_dropout"
    load_model_from_cpd([],cpd)


