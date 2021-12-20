import torch

from torch.autograd  import  Function

class ACL(torch.nn.Module):
    def __init__(self,queryvectordim, layers,activations,contextvectordim):
        super(ACL, self).__init__()

        layers=layers+[contextvectordim]

        assert len(layers)==len(activations),"number of activations must be one more than number of layers because the output layer also needs a activation which is usally set to none"
        self.linears = torch.nn.ModuleList()
        p_dim=queryvectordim
        for i in range(len(layers)):
            self.linears.append(torch.nn.Linear(p_dim,int(layers[i])))
            p_dim=int(layers[i])
            if activations[i]=="relu":
                self.linears.append(torch.nn.ReLU())
            elif activations[i]=="sigmoid":
                self.linears.append(torch.nn.Sigmoid())
            elif activations[i]=="none":
                pass



    def forward(self,queryvector,yi):  #所有ilme方法的forward参数需要保持一致 yi是为 minilstm保留的
        x=queryvector
        for l in self.linears:
            x=l(x)
        return x


class NACL(torch.nn.Module):

    def __init__(self, dim):
        super(NACL, self).__init__()
        self.psudo_ctxvector = torch.nn.Parameter(torch.empty((dim)))
    def forward(self, queryvector,yi):
        return self.psudo_ctxvector




class RevGrad(torch.nn.Module):
    def __init__(self, alpha=1.0, *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)
        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return RevGrad_F.apply(input_, self._alpha)
class RevGrad_F(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None




if __name__=="__main__":
    import numpy as np
    import random
    seed=0
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.


    testmodel1=ACL(256,"512_512","relu_relu_none",256)
    grl=RevGrad()
    testmodel2=ACL(256,"512_512","relu_relu_none",1)
    ilme_parameter=list(testmodel1.parameters())

    test_in=torch.rand(256)
    testout1=testmodel1(test_in,0)
    testout1=grl(testout1)
    testout=testmodel2(testout1,0)
    loss=100*testout.square().mean()
    loss.backward()

    ilme_parameter[0].register_hook(lambda grad: grad * -2)  # double the gradient

    testout1=testmodel1(test_in,0)
    testout1=grl(testout1)
    testout=testmodel2(testout1,0)
    loss=100*testout.square().mean()
    loss.backward()



    with torch.no_grad():
        print(ilme_parameter[0].grad.mean(),ilme_parameter[0].grad.std(),torch.abs(ilme_parameter[0].grad).mean())
        debug=np.array(ilme_parameter[0].grad)
    b=1