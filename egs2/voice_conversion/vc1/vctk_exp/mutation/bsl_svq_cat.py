from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from espnet2.VC_SRC.Model_component.layers import ConvNorm, LinearNorm,Tacotron1_prenet,concate_condition
from espnet2.VC_SRC.Model_component.VC_utils import  get_mask_from_lengths
from espnet2.VC_SRC.Model_component.default_hparams_nonparallel import  create_hparams
#from espnet2.VC_SRC.Model_component.VC_utils import  qk2w


#来自rfs_dp_3gru_simatt_cat_res256_t_fix_cat_prenetbias_embinit_Fc_norsp_nodp
#使用soft vector quantization 进行隔离
datadir='/home/projects/12001458/yufei/espnet/egs2/voice_conversion/vc1/vctk_exp/data_20spk'

optimizer=torch.optim.Adam
def getlr(step, lr0=0.001,warmup_steps = 4000.0):
    lr=lr0 * warmup_steps**0.5 * min(step * warmup_steps**-1.5, step**-0.5)
    return lr


hparams=create_hparams()

#hparam override
hparams.encoder_embedding_dim=512
hparams.decoder_rnn_dim=256
hparams.prenet_depths=[512,512,512]



hparams.dictionary_size=32   #字典的大小



class VC_model(nn.Module):
    def __init__(self, hparams):
        super(VC_model, self).__init__()

        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step

        self.prenet=Tacotron1_prenet(hparams.BN_feat_dim,[hparams.prenet_dim])

        self.soft_vq=SoftVQ(hparams.dictionary_size,hparams.encoder_embedding_dim,hparams.encoder_embedding_dim)

        self.speaker_condition_layer=concate_condition(hparams.num_of_spk,hparams.speaker_emb_dim)

        self.dim_reduction_layer=torch.nn.Linear(hparams.prenet_dim+hparams.speaker_emb_dim,hparams.encoder_embedding_dim, bias=True)

        hparams.encoder_embedding_dim=hparams.encoder_embedding_dim
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        self.postnet_Fc=torch.nn.Linear(hparams.n_mel_channels,hparams.n_mel_channels, bias=True)
        self.loss=Tacotron2Loss()


    def forward(self, BN_feat,seq_length,speaker_id,mel_target):

        encoder_output=self.prenet(BN_feat)
        weight,vq_encoder_output=self.soft_vq(encoder_output)
        encoder_output=self.speaker_condition_layer(vq_encoder_output,speaker_id)

        encoder_output=self.dim_reduction_layer(encoder_output)


        mel_outputs_mstep = self.decoder(
            encoder_output, mel_target, memory_lengths=seq_length)
        mel_outputs = mel_outputs_mstep.view(
            mel_outputs_mstep.size(0), -1, self.n_mel_channels)

        mel_outputs_postnet = self.postnet(mel_outputs_mstep.transpose(1, 2))
        mel_outputs_postnet = mel_outputs_mstep + mel_outputs_postnet.transpose(1, 2)



        mel_outputs_postnet = mel_outputs_postnet.view(
            mel_outputs_postnet.size(0), -1, self.n_mel_channels)


        mel_outputs_postnet=self.postnet_Fc(mel_outputs_postnet)
        mel_loss,mel_loss_final=self.loss(mel_outputs,mel_outputs_postnet,mel_target)
        total_loss=mel_loss+mel_loss_final
        state={}
        state["mel_loss"]=mel_loss
        state["mel_loss_final"]=mel_loss_final
        state["hist"]= {}
        state["hist"]["weight"]= weight

        return total_loss,mel_outputs_postnet,state




    def inference(self, BN_feat,seq_length,speaker_id):

        encoder_output=self.prenet(BN_feat)
        weight,vq_encoder_output=self.soft_vq(encoder_output)
        encoder_output=self.speaker_condition_layer(vq_encoder_output,speaker_id)
        encoder_output=self.dim_reduction_layer(encoder_output)
        mel_outputs_mstep = self.decoder.inference(
            encoder_output)


        mel_outputs = mel_outputs_mstep.view(
            mel_outputs_mstep.size(0), -1, self.n_mel_channels)

        mel_outputs_postnet = self.postnet(mel_outputs_mstep.transpose(1, 2))
        mel_outputs_postnet = mel_outputs_mstep + mel_outputs_postnet.transpose(1, 2)



        mel_outputs_postnet = mel_outputs_postnet.view(
            mel_outputs_postnet.size(0), -1, self.n_mel_channels)



        mel_outputs_postnet=self.postnet_Fc(mel_outputs_postnet)

        return mel_outputs_postnet,0






class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim


        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps


        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.n_mel_channels,
            hparams.prenet_depths)



        self.decoder_rnn1_dim = hparams.attention_rnn_dim
        self.decoder_rnn1 = nn.GRUCell(
            hparams.prenet_depths[-1]+hparams.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        self.att_project = LinearNorm(hparams.attention_rnn_dim+hparams.encoder_embedding_dim,
            hparams.decoder_rnn_dim)

        self.decoder_rnn2_dim = hparams.decoder_rnn_dim

        self.decoder_rnn2 = nn.GRUCell(
            hparams.decoder_rnn_dim,
            hparams.decoder_rnn_dim, True)

        self.decoder_rnn3_dim = hparams.decoder_rnn_dim

        self.decoder_rnn3 = nn.GRUCell(
            hparams.decoder_rnn_dim,
            hparams.decoder_rnn_dim, True)



        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim+self.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)


    def decode(self, decoder_input,memory):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        decoder_input = torch.cat(
            (decoder_input, memory), -1)



        self.decoder_hidden1 = self.decoder_rnn1(
            decoder_input, self.decoder_hidden1)




        decoder2_in=self.att_project(torch.cat(
            (self.decoder_hidden1, memory), -1))

        self.decoder_hidden2 = self.decoder_rnn2(
            decoder2_in, self.decoder_hidden2)

        res_decoder_hidden2=self.decoder_hidden2+decoder2_in


        self.decoder_hidden3 = self.decoder_rnn3(
            res_decoder_hidden2, self.decoder_hidden3)

        res_decoder_hidden3=self.decoder_hidden3+res_decoder_hidden2

        decode_concate = torch.cat(
            (res_decoder_hidden3, memory), dim=1)
        decoder_output = self.linear_projection(decode_concate)


        return decoder_output


    def forward(self, memorys, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs   #mel 答案
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memorys)


        decoder_inputs=decoder_inputs.transpose(0,1)


        decoder_inputs=decoder_inputs[ self.n_frames_per_step - 1::self.n_frames_per_step, :, : ]
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)[0:-1]  #前面粘另一个全零的向量
        decoder_inputs = self.prenet(decoder_inputs)


        memorys = torch.unsqueeze(memorys, -1)

        memorys = memorys.reshape(memorys.size(0), -1, self.n_frames_per_step, memorys.size(2))
        memorys = memorys.mean(2)   #模拟attention


        memorys=memorys.transpose(0,1)

        self.initialize_decoder_states(
            memorys, mask=~get_mask_from_lengths(memory_lengths))




        mel_outputs = []
        while len(mel_outputs) < decoder_inputs.size(0) :
            decoder_input = decoder_inputs[len(mel_outputs)]
            memory=memorys[len(mel_outputs)]
            mel_output = self.decode(
                decoder_input,memory)
            mel_outputs += [mel_output.squeeze(1)]

        mel_outputs = self.parse_decoder_outputs(
            mel_outputs)


        return mel_outputs

    def inference(self, memorys):
        decoder_input = self.get_go_frame(memorys)[0,:,:]


        memorys = torch.unsqueeze(memorys, -1)
        memorys = memorys.reshape(memorys.size(0), -1, self.n_frames_per_step, memorys.size(2))
        memorys = memorys.mean(2)
        memorys = memorys.transpose(0, 1)

        self.initialize_decoder_states(memorys, mask=None)

        mel_outputs = [ ]
        while True:

            decoder_input = self.prenet(decoder_input)
            mel_output = self.decode(decoder_input, memorys[ len(mel_outputs) ])
            mel_outputs += [ mel_output.squeeze(1) ]

            if len(mel_outputs) > memorys.shape[ 0 ] - 1:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output[ :, -self.n_mel_channels: ]

        mel_outputs = self.parse_decoder_outputs(
            mel_outputs)

        return mel_outputs




    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(self.n_frames_per_step-1,
            B, self.n_mel_channels ).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(1)
        MAX_TIME = memory.size(0)

        self.decoder_hidden1 = Variable(memory.data.new(
            B, self.decoder_rnn1_dim).zero_())
        self.decoder_cell1 = Variable(memory.data.new(
            B, self.decoder_rnn1_dim).zero_())



        self.decoder_hidden2 = Variable(memory.data.new(
            B, self.decoder_rnn2_dim).zero_())
        self.decoder_cell2 = Variable(memory.data.new(
            B, self.decoder_rnn2_dim).zero_())

        self.decoder_hidden3 = Variable(memory.data.new(
            B, self.decoder_rnn2_dim).zero_())
        self.decoder_cell3 = Variable(memory.data.new(
            B, self.decoder_rnn2_dim).zero_())

        self.memory = memory
        self.mask = mask



    def parse_decoder_outputs(self, mel_outputs):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """


        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()

        return mel_outputs





class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=True)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=self.training)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels*hparams.n_frames_per_step, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels*hparams.n_frames_per_step,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels*hparams.n_frames_per_step))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.l1loss=nn.L1Loss(reduction="mean")




    def forward(self, mel_out,mel_out_postnet, mel_target):

        mel_target.requires_grad = False

        mel_loss = self.l1loss(mel_out, mel_target)

        mel_loss_final=self.l1loss(mel_out_postnet, mel_target)

        return mel_loss,mel_loss_final






class SoftVQ(torch.nn.Module):


    def __init__(self,dictionary_size,key_dim,dictionary_dim):
        super(SoftVQ, self).__init__()


        self.keys=torch.nn.Parameter(torch.randn((key_dim,dictionary_size)))         # key(dim,dictionary_size)
        self.value=torch.nn.Parameter(torch.randn((dictionary_size,dictionary_dim)))         # key(dictionary_size,dim)


    def forward(self,query):
        # 利用query和key计算权重，使用dotproduct
        # query(batch_size,length,dim)

        weight = torch.softmax(torch.matmul(query, self.keys),dim=2)    #TODO 是否需要像transformer中一样/dimk
                                                    # weight(batch,length,dictionary_size)
        VQed_output = torch.matmul(weight, self.value)  #

        return weight,VQed_output






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
    mel_loss,mel_outputs_postnet,state = m.forward(bn_feat, seq_length,spk_id,target_mel)
    #out,cc = m.inference(bn_feat, seq_length,spk_id)

    print(mel_outputs_postnet.mean(),mel_outputs_postnet.std())

