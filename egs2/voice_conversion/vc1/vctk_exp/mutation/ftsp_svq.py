from espnet2.VC_SRC.Model_component.Fastspeech2.transformer import  Decoder, PostNet
from espnet2.VC_SRC.Model_component.Fastspeech2.default_param import fastspeech_model_config
from torch import nn
import torch
from espnet2.VC_SRC.Model_component.default_hparams_nonparallel import  create_hparams
from espnet2.VC_SRC.Model_component.layers import Tacotron1_prenet,attention_contition
from espnet2.VC_SRC.Model_component.Fastspeech2.utils import get_mask_from_lengths
from espnet2.VC_SRC.Model_component.Fastspeech2.Loss import FastSpeech2Loss
import math
hparams=create_hparams()


datadir='/home/projects/12001458/yufei/espnet/egs2/voice_conversion/vc1/vctk_exp/data_20spk'
optimizer=torch.optim.Adam
def getlr(step,warm_up_step=4000):
    init_lr = np.power(fastspeech_model_config["transformer"]["decoder_hidden"], -0.5)
    lr = np.min(
        [
            np.power(step, -0.5),
            np.power(warm_up_step, -1.5) * step,
        ]
    )
    return init_lr*lr

#parameter override
#model_config
hparams.batchsize=64
hparams.dictionary_size=32   #字典的大小
hparams.speaker_emb_dim=16   #spk embedding 的维度  dictionary_size*speaker_emb_dim 才是spk emb的真实维度

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self):
        super(FastSpeech2, self).__init__()
        self.model_config = fastspeech_model_config


        self.decoder = Decoder(fastspeech_model_config)
        self.mel_linear = nn.Linear(
            fastspeech_model_config["transformer"]["decoder_hidden"],
            hparams.n_mel_channels,
        )
        self.postnet = PostNet()


    def forward(
        self,
        input_feats,
        src_lens,
    ):
        src_masks = get_mask_from_lengths(src_lens, None)

        output, mel_masks = self.decoder(input_feats, src_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            mel_masks
        )





class SoftVQ(torch.nn.Module):


    def __init__(self,dictionary_size,key_dim,dictionary_dim):
        super(SoftVQ, self).__init__()


        self.keys=torch.nn.Parameter(torch.randn((key_dim,dictionary_size)))         # key(dim,dictionary_size)
        self.value=torch.nn.Parameter(torch.randn((dictionary_size,dictionary_dim)))         # key(dictionary_size,dim)
        self.sqrt_d_k=math.sqrt(key_dim)

    def forward(self,query,norm_weight=True):
        # 利用query和key计算权重，使用dotproduct
        # query(batch_size,length,dim)
        if norm_weight:
            weight=torch.matmul(query, self.keys)/self.sqrt_d_k
        else:
            weight = torch.matmul(query, self.keys)
        weight = torch.softmax(weight,dim=2)
                                                    # weight(batch,length,dictionary_size)
        VQed_output = torch.matmul(weight, self.value)  #

        return weight,VQed_output


class VC_model(nn.Module):
    def __init__(self, hparams):
        super(VC_model, self).__init__()

        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step

        self.prenet=Tacotron1_prenet(hparams.BN_feat_dim,[hparams.prenet_dim])
        self.soft_vq = SoftVQ(hparams.dictionary_size, hparams.prenet_dim, hparams.prenet_dim)
        self.speaker_condition_layer=attention_contition(hparams.num_of_spk,hparams.dictionary_size,hparams.speaker_emb_dim)

        self.dim_reduction_layer=torch.nn.Linear(hparams.prenet_dim+hparams.speaker_emb_dim,fastspeech_model_config["transformer"]["decoder_hidden"], bias=True)

        hparams.encoder_embedding_dim=hparams.encoder_embedding_dim
        self.fastspeech = FastSpeech2()

        self.loss=FastSpeech2Loss()


    def forward(self, BN_feat,seq_length,speaker_id,mel_target):

        encoder_output=self.prenet(BN_feat)
        weight, vq_encoder_output = self.soft_vq(encoder_output)
        encoder_output=self.speaker_condition_layer(vq_encoder_output,weight,speaker_id)

        encoder_output=self.dim_reduction_layer(encoder_output)


        output,postnet_output,mel_masks= self.fastspeech(encoder_output,seq_length)

        mel_loss,postnet_mel_loss= self.loss(output,postnet_output,mel_target,mel_masks)


        total_loss=mel_loss+postnet_mel_loss
        state={}
        state["mel_loss"]=mel_loss
        state["mel_loss_final"]=postnet_mel_loss
        return total_loss,postnet_output,state




    def inference(self, BN_feat,seq_length,speaker_id):

        encoder_output=self.prenet(BN_feat)
        weight, vq_encoder_output = self.soft_vq(encoder_output)
        encoder_output=self.speaker_condition_layer(vq_encoder_output,weight,speaker_id)

        encoder_output=self.dim_reduction_layer(encoder_output)


        output,postnet_output,_= self.fastspeech(encoder_output,seq_length)

        return postnet_output,0


if __name__ == '__main__':
    import numpy as np

    m = VC_model(hparams)
    m.eval()
    B=4
    max_length=22
    bn_feat=torch.rand((B,max_length,512))
    target_mel=torch.ones((B,max_length,80))
    seq_length=torch.randint(0,max_length,(B,))
    spk_id=torch.randint(0,22,(B,))

    seq_length[0]=max_length
    loss_total,mel_final,state = m.forward(bn_feat, seq_length,spk_id,target_mel)
    out,cc = m.inference(bn_feat, seq_length,spk_id)

    print(mel_final.mean(),mel_final.std())
    print(state)
