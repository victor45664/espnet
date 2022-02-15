from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F
from typeguard import check_argument_types
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.lm.abs_model import AbsLM
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss_kd

class ESPnetLanguageModel(AbsESPnetModel):
    def __init__(self, lm: AbsLM, vocab_size: int, ignore_id: int = 0):
        assert check_argument_types()
        super().__init__()
        self.lm = lm
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1

        # ignore_id may be assumed as 0, shared with CTC-blank symbol for ASR.
        self.ignore_id = ignore_id

    def nll(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute negative log likelihood(nll)

        Normally, this function is called in batchify_nll.
        Args:
            text: (Batch, Length)
            text_lengths: (Batch,)
            max_lengths: int
        """
        batch_size = text.size(0)
        # For data parallel
        if max_length is None:
            text = text[:, : text_lengths.max()]
        else:
            text = text[:, :max_length]

        # 1. Create a sentence pair like '<sos> w1 w2 w3' and 'w1 w2 w3 <eos>'
        # text: (Batch, Length) -> x, y: (Batch, Length + 1)
        x = F.pad(text, [1, 0], "constant", self.eos)
        t = F.pad(text, [0, 1], "constant", self.ignore_id)
        for i, l in enumerate(text_lengths):
            t[i, l] = self.sos
        x_lengths = text_lengths + 1

        # 2. Forward Language model
        # x: (Batch, Length) -> y: (Batch, Length, NVocab)
        y, _ = self.lm(x, None)

        # 3. Calc negative log likelihood
        # nll: (BxL,)
        nll = F.cross_entropy(y.view(-1, y.shape[-1]), t.view(-1), reduction="none")
        # nll: (BxL,) -> (BxL,)
        if max_length is None:
            nll.masked_fill_(make_pad_mask(x_lengths).to(nll.device).view(-1), 0.0)
        else:
            nll.masked_fill_(
                make_pad_mask(x_lengths, maxlen=max_length + 1).to(nll.device).view(-1),
                0.0,
            )
        # nll: (BxL,) -> (B, L)
        nll = nll.view(batch_size, -1)
        return nll, x_lengths

    def batchify_nll(
        self, text: torch.Tensor, text_lengths: torch.Tensor, batch_size: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute negative log likelihood(nll) from transformer language model

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            text: (Batch, Length)
            text_lengths: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase

        """
        total_num = text.size(0)
        if total_num <= batch_size:
            nll, x_lengths = self.nll(text, text_lengths)
        else:
            nlls = []
            x_lengths = []
            max_length = text_lengths.max()

            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_text = text[start_idx:end_idx, :]
                batch_text_lengths = text_lengths[start_idx:end_idx]
                # batch_nll: [B * T]
                batch_nll, batch_x_lengths = self.nll(
                    batch_text, batch_text_lengths, max_length=max_length
                )
                nlls.append(batch_nll)
                x_lengths.append(batch_x_lengths)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nlls)
            x_lengths = torch.cat(x_lengths)
        assert nll.size(0) == total_num
        assert x_lengths.size(0) == total_num
        return nll, x_lengths

    def forward(
        self, text: torch.Tensor, text_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        nll, y_lengths = self.nll(text, text_lengths)
        ntokens = y_lengths.sum()
        loss = nll.sum() / ntokens
        stats = dict(loss=loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, ntokens), loss.device)
        return loss, stats, weight

    def collect_feats(
        self, text: torch.Tensor, text_lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {}



class ESPnetLanguageModel_kd(ESPnetLanguageModel):

    def __init__(self, lm: AbsLM,ilm_for_kd: torch.nn.Module, vocab_size: int, ignore_id: int = 0):
        assert check_argument_types()
        super(ESPnetLanguageModel, self).__init__()
        self.lm = lm
        self.lm_parameters=self.parameters()  #parameters in lm model
        self.decoder=ilm_for_kd   #这里加decoder是为了方便 从asr模型的参数中初始化
        self.decoder.eval()       #ILM 应该始终保持eval模式
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1

        # ignore_id may be assumed as 0, shared with CTC-blank symbol for ASR.
        self.ignore_id = ignore_id
        self.kl_loss=torch.nn.KLDivLoss(reduction="none")
    def nll(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        max_length: Optional[int] = None,doing_kd=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute negative log likelihood(nll)

        Normally, this function is called in batchify_nll.
        Args:
            text: (Batch, Length)
            text_lengths: (Batch,)
            max_lengths: int
        """
        batch_size = text.size(0)
        # For data parallel
        if max_length is None:
            text = text[:, : text_lengths.max()]
        else:
            text = text[:, :max_length]

        # 1. Create a sentence pair like '<sos> w1 w2 w3' and 'w1 w2 w3 <eos>'
        # text: (Batch, Length) -> x, y: (Batch, Length + 1)
        x = F.pad(text, [1, 0], "constant", self.eos)
        t = F.pad(text, [0, 1], "constant", self.ignore_id)
        for i, l in enumerate(text_lengths):
            t[i, l] = self.sos
        x_lengths = text_lengths + 1

        # 2. Forward Language model
        # x: (Batch, Length) -> y: (Batch, Length, NVocab)
        y, _ = self.lm(x, None)


        # 4. Calc negative log likelihood
        # nll: (BxL,)
        nll = F.cross_entropy(y.view(-1, y.shape[-1]), t.view(-1), reduction="none")


        # nll: (BxL,) -> (BxL,)
        if max_length is None:
            nll.masked_fill_(make_pad_mask(x_lengths).to(nll.device).view(-1), 0.0)
        else:
            nll.masked_fill_(
                make_pad_mask(x_lengths, maxlen=max_length + 1).to(nll.device).view(-1),
                0.0,
            )
        # nll: (BxL,) -> (B, L)
        nll = nll.view(batch_size, -1)
        lm_acc = th_accuracy(
            y.view(-1, self.vocab_size),
            t,
            ignore_label=self.ignore_id,
        )
        if doing_kd:
            # 3. Forward internal Language model
            # x: (Batch, Length) -> y: (Batch, Length, NVocab)

            with torch.no_grad():
                fake_encoder_out = x.new_zeros(batch_size, 1, self.encoder_output_size)
                self.decoder.eval()
                ilm_output, _ = self.decoder.forward_ilm(
                    fake_encoder_out, -1, x, x_lengths
                )




                ilm_output = torch.log_softmax(ilm_output, dim=2).detach()
                #vdebug torch.sum(torch.exp(ilm_output),dim=2)==1
                true_dist = y.clone()
                true_dist = true_dist.fill_(self.label_smooth / (self.vocab_size - 1))
                ignore = t == self.ignore_id
                target = t.masked_fill(ignore, 0)  # avoid -1 index
                true_dist.scatter_(2, target.unsqueeze(2), 1-self.label_smooth)

            lm_log_softmax=torch.log_softmax(y, dim=2)
            lm_ilm_output=lm_log_softmax+self.kd_factor*ilm_output
            kd_loss = self.kl_loss(torch.log_softmax(lm_ilm_output, dim=2).view(-1, self.vocab_size), true_dist.view(-1, self.vocab_size))
            kd_loss=torch.sum(kd_loss,dim=1)
            lm_ilm_acc = th_accuracy(
                lm_ilm_output.view(-1, self.vocab_size),
                t,
                ignore_label=self.ignore_id,
            )
            # kd_loss: (BxL,) -> (BxL,)
            if max_length is None:
                kd_loss.masked_fill_(make_pad_mask(x_lengths).to(kd_loss.device).view(-1), 0.0)
            else:
                kd_loss.masked_fill_(
                    make_pad_mask(x_lengths, maxlen=max_length + 1).to(kd_loss.device).view(-1),
                    0.0,
                )
            kd_loss = kd_loss.view(batch_size, -1)
            with torch.no_grad():
                # 4. Calc negative log likelihood
                # nll: (BxL,)
                kd_nll = F.cross_entropy(lm_ilm_output.view(-1, y.shape[-1]), t.view(-1), reduction="none")

                # nll: (BxL,) -> (BxL,)
                if max_length is None:
                    kd_nll.masked_fill_(make_pad_mask(x_lengths).to(kd_nll.device).view(-1), 0.0)
                else:
                    kd_nll.masked_fill_(
                        make_pad_mask(x_lengths, maxlen=max_length + 1).to(kd_nll.device).view(-1),
                        0.0,
                    )
                # nll: (BxL,) -> (B, L)
                kd_nll = kd_nll.view(batch_size, -1)



            return nll,lm_acc,kd_nll,lm_ilm_acc,kd_loss,x_lengths

        else:
            return nll, x_lengths


    def forward(
        self, text: torch.Tensor, text_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        nll,lm_acc,kd_nll,lm_ilm_acc,kd_loss, y_lengths = self.nll(text, text_lengths,doing_kd=True)
        ntokens = y_lengths.sum()
        nll = nll.sum() / ntokens
        kd_nll = kd_nll.sum() / ntokens
        kd_loss = kd_loss.sum() / ntokens
        stats = dict(
            nll=nll.detach(),
            kd_nll=kd_nll.detach(),
            kd_loss=kd_loss.detach(),
            lm_ilm_acc=lm_ilm_acc,
            lm_acc=lm_acc,
                     )
        loss=kd_loss+1
        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, ntokens), loss.device)
        return loss, stats, weight