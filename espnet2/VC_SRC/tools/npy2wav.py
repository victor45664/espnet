# -*- coding: utf-8 -*-
import sys
sys.path.append('')
import os
from espnet2.VC_SRC import gen_wavernn as vocoder
from shutil import rmtree


def npy2wav(npy_dir,out_wav_dir):
    #这个函数遍历一个目录下面所有npy文件，并且用对应的vocoder全部转换成wav

    rmtree(out_wav_dir,ignore_errors=True)
    os.makedirs(out_wav_dir,exist_ok=True)
    for root, _, files in os.walk(npy_dir):
        dir_relative_path = os.path.relpath(root, npy_dir)
        for f in files:
            if ".npy" in f:   #发现npy就生成wav
                os.makedirs(os.path.join(out_wav_dir,dir_relative_path), exist_ok=True)
                vocoder(os.path.join(npy_dir,dir_relative_path, f),os.path.join(out_wav_dir,dir_relative_path,f[:-4]+'.wav'))




if __name__ == '__main__':

    npy_dir=r'F:\temp\新建文件夹'
    out_wav_dir=r'F:\temp\新建文件夹_wavernn'
    npy2wav(npy_dir, out_wav_dir)





