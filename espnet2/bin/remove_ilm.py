
import torch
import sys


pth_path=sys.argv[1]

cont = torch.load(pth_path,map_location='cpu')
ilme_key=[]
for key in cont.keys():
    if "lm." not in key:
        print(key)
        ilme_key.append(key)


for key in ilme_key:
    del cont[key]
torch.save(cont,pth_path+"_noilm")






