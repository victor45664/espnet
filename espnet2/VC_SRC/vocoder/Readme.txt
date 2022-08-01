示例：
python vocoder.py --path_input_mels /home/mas-liu.yufei/vc/voice_conversion/models/VCTK/eval_result/baseline_mvemb_larger_larger/19/p376_020.npy --path_output_dir ./generated_wavs --checkpoint_file vocoder

--path_input_mels: 输入梅尔谱的路径，可以为文件夹或单文件
--path_output_dir：输出生成语音的路径
--checkpoint_file：模型路径