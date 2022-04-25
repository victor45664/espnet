# -*- coding: utf-8 -*-

import os
import sys

import pathlib


def path_to_model_save_path(path):  #从module所在的路径计算出模型保存的危指


    mutation = os.path.basename(path).split('.')[0]

    model_save_path=os.path.join(path, '../../..', 'newest_model_saved', mutation)


    return model_save_path




def path_to_module(path,dir_level):  #dir_level是该module所在的更目录在多少级以上

    temp=path.split(r'/')
    module_str=''

    for i in range(len(temp)-dir_level-1,len(temp)-1):
        module_str+=temp[i]+'.'
    module_str += temp[-1].split('.')[0]




    return module_str  #输入path返回可以被动态import的字符串




def dynamic_import_from_abs_path(mutation_path):

    cwd_path = pathlib.Path(mutation_path)
    cwd_path=os.path.join(cwd_path.parent, "../../../..")  #项目的根目录总是再mutation上四级
    sys.path.append(cwd_path)


    mutation_path=mutation_path.replace(r'//',r'/')  #处理一些拼接错误，防止出现两个/


    converted_str=path_to_module(mutation_path,3)
    #converted_str=r'models.baseline_fbank.mutation.baseline_k_001_larger_hl_adam_deeper_nar_deeper'


    imported_module=__import__(converted_str,fromlist=True)  #全局动态import


    sys.path.remove(cwd_path)  #import完毕后将path删除防止影响 当前代码的正常运行

    return imported_module



if __name__ == '__main__':
    # vc_model_mutation_path = '$root_path/tf_project/a2m_VC/models/single_spk/mutation/baseline_with_emb.py'
    # asr_model_mutation_path = '$root_path/tf_project/tensorflow_ASR/models/ASR_mel80/mutation/baseline_with_emb.py'
    # sv_model_mutation_path = '$root_path/tf_project/tensorflow_ASR/models/SV_mel80/mutation/baseline_with_emb.py'
    #
    #
    # asr_model=dynamic_import_from_abs_path(r'E:/python_project/tensorflow_ASR/models/SV_mel80/mutation/baseline.py')
    print(path_to_model_save_path('./models/VCTK_mt/mutation/baseline_mvemb_larger_larger2.py'))


