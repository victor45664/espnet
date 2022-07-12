import torch

from espnet2.VC_SRC.Model_component.VC_utils import get_mask_from_lengths








class Tacotron1_prenet(torch.nn.Module):
    def __init__(self, input_dim,layer_sizes, dropout=0.5):
        super(Tacotron1_prenet, self).__init__()
        self.dropout_rate=dropout
        self.layers=[]
        self.layers.append(LinearNorm(input_dim,layer_sizes[0]))
        for i in range(1,len(layer_sizes)):
            self.layers.append(LinearNorm(layer_sizes[i-1],layer_sizes[i]))

            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Dropout(dropout))

        self._forward=torch.nn.Sequential(*self.layers)


    def forward(self,x):

        x=self._forward(x)

        return x

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)   #[B,2,T]
        return conv_signal
        
class attention_contition(torch.nn.Module):
    def __init__(self, num_of_spk,dictionary_size, emb_dim):
        super(attention_contition, self).__init__()
        self.speaker_embedding = torch.nn.Embedding(num_of_spk, emb_dim*dictionary_size)
        self.emb_dim=emb_dim
        self.dictionary_size=dictionary_size

    def forward(self, x,weight, speaker_id):
        spkemb = self.speaker_embedding(speaker_id)


        atten_spkemb = torch.matmul(weight, spkemb.view(-1, self.dictionary_size, self.emb_dim))
        conditioned_x = torch.cat([x, atten_spkemb], dim=2)


        return conditioned_x


class concate_condition(torch.nn.Module):
    def __init__(self,num_of_spk,emb_dim):
        super(concate_condition, self).__init__()
        self.speaker_embedding=torch.nn.Embedding(num_of_spk, emb_dim)


    def forward(self,x,speaker_id):
        emb=self.speaker_embedding(speaker_id)
        emb=emb.unsqueeze(1)
        expanded_emb = emb.expand(emb.shape[0],x.shape[1],emb.shape[2])
        concated=torch.cat([x,expanded_emb],dim=2)
        return concated

class mvemb_condition(torch.nn.Module):
    def __init__(self,num_of_spk,emb_dim):
        super(mvemb_condition, self).__init__()
        self.speaker_embedding=torch.nn.Embedding(num_of_spk, emb_dim*2)
        self.emb_dim=emb_dim
        self.sp=StatisticPool()

    def forward(self,x,seq_length,speaker_id):
        emb=self.speaker_embedding(speaker_id)
        mean_emb=emb[:,:self.emb_dim]
        std_emb=emb[:,self.emb_dim:]

        s_mean,s_std=self.sp(x,seq_length)
        norm_x=(x-s_mean.unsqueeze(1))/s_std.unsqueeze(1)
        conditioned_x=norm_x*std_emb.unsqueeze(1)+mean_emb.unsqueeze(1)


        return conditioned_x


class StatisticPool(torch.nn.Module):
    def __init__(self,EPSILON=1e-5):
        super(StatisticPool, self).__init__()
        self.EPSILON=EPSILON

    def forward(self,x,seq_length):    #x[B,T,D]
        D=x.size(2)
        mask=get_mask_from_lengths(seq_length)
        mask = mask.unsqueeze(2) # [N, T,1]

        mask = mask.expand(mask.size(0), mask.size(1),D)
        x=x*mask

        x_sum=torch.sum(x,dim=1).transpose(0,1)
        x_mean=(x_sum/seq_length).transpose(0,1)

        x_cut_mean=x-x_mean.unsqueeze(1)
        x_cut_mean=x_cut_mean*mask

        x_var=torch.square(x_cut_mean).sum(1).transpose(0,1)
        x_std = torch.sqrt((x_var / seq_length).transpose(0, 1)+self.EPSILON)

        return x_mean,x_std  # [B,D]





if __name__ == '__main__':
    D=3
    sp=mvemb_condition(11,D)

    B=2
    max_length=7
    spk_id=torch.randint(0,11,(B,))

    # x=np.ones(((B,max_length,D)))
    # #x[2,2,2]=11
    # x=torch.from_numpy(x)
    x=torch.rand((B,max_length,D))
    seq=torch.randint(0,max_length,(B,))
    seq[0]=max_length
    mean,std=sp(x,seq,spk_id)
    b=1














