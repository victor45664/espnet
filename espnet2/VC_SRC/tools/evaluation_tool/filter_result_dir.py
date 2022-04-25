

from espnet2.VC_SRC import melspectrogram,load_wav

import sys
sys.path.append('')
import numpy as np
import os
from shutil import copyfile,rmtree

#对比不同模型生成的wav的差异
# 筛选差异比较大的wav给人工评价，节省人工时间


def cal_npy_diff(npy1,npy2):
    sq_diff=np.square((npy1-npy2))
    sum_diff=np.mean(sq_diff)
    diff=np.sqrt(sum_diff)

    return diff


def cal_multi_matrix_std(mats):  #计算多个npy矩阵的std
    stacked_mats=np.stack(mats,axis=0)
    std=stacked_mats.std(axis=0)


    return std.mean()


def wav_to_mel_spec(wav_path): #计算mel谱子，用于计算wav差异度

    print("cal")
    wav = load_wav(wav_path)
    mel_spectrogram = melspectrogram(wav).astype(np.float32)  #没有mel谱，重新计算


    return mel_spectrogram



def compare_with_npy(dir1, dir2, output_dir, diff_threshold=0.056):    #这个函数对比eval_result下的两个文件夹中mel谱的相似度，剔除mel谱相似度高的，只留下mel谱相似度较低的，用于提高人工对比的效率

    rmtree(output_dir,ignore_errors=True)
    os.makedirs(output_dir)

    for root, dirs, files in os.walk(dir1):  #dir1和dir2中应该要有相同的结构和相同的文件
        for f in files:
            if ".wav" in f:   #按照默认格式，每一个wav边上都会有一个同名的npy文件,而这个npy文件就是生成wav用的mel谱
                dir_relative_path=os.path.relpath(root,dir1)
                npy1_path=os.path.join(dir1,dir_relative_path, f[:-4]+'.npy')
                npy2_path=os.path.join(dir2,dir_relative_path, f[:-4]+'.npy')
                npy1=np.load(npy1_path)
                npy2=np.load(npy2_path)
                diff=cal_npy_diff(npy1,npy2)
                print(diff)
                if (diff>=diff_threshold):  #差异比较大的，提取到output_dir中
                    os.makedirs(os.path.join(output_dir,os.path.basename(dir1),dir_relative_path),exist_ok=True)
                    os.makedirs(os.path.join(output_dir,os.path.basename(dir2),dir_relative_path),exist_ok=True)
                    copyfile(os.path.join(dir1,dir_relative_path, f),os.path.join(output_dir,os.path.basename(dir1),dir_relative_path, f))
                    copyfile(os.path.join(dir2,dir_relative_path, f),os.path.join(output_dir,os.path.basename(dir2),dir_relative_path, f))
                    copyfile(os.path.join(dir1,dir_relative_path, f[:-4]+'.npy'),os.path.join(output_dir,os.path.basename(dir1),dir_relative_path, f[:-4]+'.npy'))
                    copyfile(os.path.join(dir2,dir_relative_path, f[:-4]+'.npy'),os.path.join(output_dir,os.path.basename(dir2),dir_relative_path, f[:-4]+'.npy'))

def compare_multi_dir_with_npy(dirs,output_dir,diff_threshold=0.019):
    #这个函数利用生成wav的mel谱子来计算对应wav的相似度
    #wav的边上要有同名的.npy文件
    rmtree(output_dir,ignore_errors=True)
    os.makedirs(output_dir,exist_ok=True)

    for root, _, files in os.walk(dirs[0]):  #所有dirs中都要有相同的结构和相同的文件
        dir_relative_path = os.path.relpath(root, dirs[0])
        for f in files:
            if ".wav" in f:   #按照默认格式，每一个wav边上都会有一个同名的npy文件,而这个npy文件就是生成wav用的mel谱
                mels=[]
                all_exist=True
                for dir in dirs:
                    if  os.path.exists(os.path.join(dir,dir_relative_path, f)) and os.path.exists(os.path.join(dir,dir_relative_path, f[:-4]+'.npy')) :
                        mels.append(np.load(os.path.join(dir,dir_relative_path, f[:-4]+'.npy')))   #直接读取npy
                    else:
                        print(os.path.join(dir,dir_relative_path, f),"does not exist,skip")
                        all_exist = False
                        break #某个文件夹中没有相同的文件，或者没有对应的npy，因此无法对比，break
                if all_exist==False:
                    continue
                diff=cal_multi_matrix_std(mels)
                print(diff)
                if (diff>=diff_threshold):  #差异比较大的，提取到output_dir中
                    for dir in dirs:
                        os.makedirs(os.path.join(output_dir,os.path.basename(dir),dir_relative_path),exist_ok=True)
                        copyfile(os.path.join(dir,dir_relative_path, f),os.path.join(output_dir,os.path.basename(dir),dir_relative_path, f))
                        copyfile(os.path.join(dir,dir_relative_path, f[:-4]+'.npy'),os.path.join(output_dir,os.path.basename(dir),dir_relative_path, f[:-4]+'.npy'))


def compare_multi_dir_with_MCD(dirs,output_dir,diff_threshold=0.059):
    #这个函数利用wav计算出来的mel谱子来对比对应wav的相似度
    #与compare_multi_dir_with_npy相比唯一的区别就是diff_threshold应该设置得不一样

    rmtree(output_dir,ignore_errors=True)
    os.makedirs(output_dir)

    for root, _, files in os.walk(dirs[0]):  #所有dirs中都要有相同的结构和相同的文件
        dir_relative_path = os.path.relpath(root, dirs[0])
        for f in files:
            if ".wav" in f:   #按照默认格式，每一个wav边上都会有一个同名的npy文件,而这个npy文件就是生成wav用的mel谱
                mels=[]
                all_exist=True
                for dir in dirs:
                    if  os.path.exists(os.path.join(dir,dir_relative_path, f)):
                        mels.append(wav_to_mel_spec(os.path.join(dir,dir_relative_path, f)))
                    else:
                        print(os.path.join(dir,dir_relative_path, f),"does not exist,skip")
                        all_exist = False
                        break #某个文件夹中没有相同的文件，因此无法对比，break
                if all_exist==False:
                    continue
                diff=cal_multi_matrix_std(mels)
                print(diff)
                if (diff>=diff_threshold):  #差异比较大的，提取到output_dir中
                    for dir in dirs:
                        os.makedirs(os.path.join(output_dir,os.path.basename(dir),dir_relative_path),exist_ok=True)
                        copyfile(os.path.join(dir,dir_relative_path, f),os.path.join(output_dir,os.path.basename(dir),dir_relative_path, f))
                        copyfile(os.path.join(dir,dir_relative_path, f[:-4]+'.npy'),os.path.join(output_dir,os.path.basename(dir),dir_relative_path, f[:-4]+'.npy'))







if __name__ == '__main__':




    import sys
    dirs=[r'F:\temp\deploy_mel80\baseline_mvemb_larger_larger_finetune',
          r'F:\temp\deploy_mel80\baseline_mvemb_larger_larger']
    output_dir=r'F:\temp\deploy_mel80\comp'
    compare_multi_dir_with_npy(dirs, output_dir,diff_threshold=0.015)










