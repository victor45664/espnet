#!/bin/bash


wav_dir=data/${spk}/wav/
out_dir=data/${spk}/

python scripts/torch_feature_extractor.py ${wav_dir} ${out_dir} 16000 1024 800 200 0 8000 80
