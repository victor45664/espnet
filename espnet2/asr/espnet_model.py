from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,LabelSmoothingLoss_kd,LabelSmoothingLoss_unadl  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from torch.distributions.categorical import Categorical
if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: AbsDecoder,
        ctc: CTC,
        rnnt_decoder: None,
        ctc_weight: float = 0.5,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert rnnt_decoder is None, "Not implemented"

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder
        # we set self.decoder = None in the CTC mode since
        # self.decoder parameters were never used and PyTorch complained
        # and threw an Exception in the multi-GPU experiment.
        # thanks Jeff Farris for pointing out the issue.
        if ctc_weight == 1.0:
            self.decoder = None
        else:
            self.decoder = decoder
        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc
        self.rnnt_decoder = rnnt_decoder
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.error_calculator = None

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        forward_ilm=False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        if forward_ilm:
            return self.forward_ilm(speech,speech_lengths,text,text_lengths)
        else:
            return self.forward_normal(speech, speech_lengths, text, text_lengths)


    def forward_normal(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # 2a. Attention-decoder branch
        if self.ctc_weight == 1.0:
            loss_att, acc_att, cer_att, wer_att = None, None, None, None
        else:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        # 2b. CTC branch
        if self.ctc_weight == 0.0:
            loss_ctc, cer_ctc = None, None
        else:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        # 2c. RNN-T branch
        if self.rnnt_decoder is not None:
            _ = self._calc_rnnt_loss(encoder_out, encoder_out_lens, text, text_lengths)

        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            acc=acc_att,
            cer=cer_att,
            wer=wer_att,
            cer_ctc=cer_ctc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def forward_ilm(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...) not nessesary it is only used to get device of tensor
            speech_lengths: (Batch, )   not nessesary it is only used to get device of tensor
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            text.shape[0]
            == text_lengths.shape[0]
        ), (text.shape, text_lengths.shape)
        batch_size = text.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        ys_in_pad, ys_out_pad = add_sos_eos(text, self.sos, self.eos, self.ignore_id)
        ys_in_lens = text_lengths + 1

        fake_encoder_out=speech.new_zeros(batch_size,1,self.encoder._output_size)
        # 1. Forward decoder
        decoder_out, _ = self.decoder.forward_ilm(
            fake_encoder_out, -1, ys_in_pad, ys_in_lens
        )

        # 2. Compute ilm loss
        loss_ilm = self.criterion_att(decoder_out, ys_out_pad)
        ilm_acc = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        loss=loss_ilm+1 #make sure
        stats = dict(
            ilm_loss=loss_ilm.detach(),
            ilm_acc=ilm_acc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def forward_ilm_validation(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...) not nessesary it is only used to get device of tensor
            speech_lengths: (Batch, )   not nessesary it is only used to get device of tensor
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
                text.shape[0]
                == text_lengths.shape[0]
        ), (text.shape, text_lengths.shape)
        batch_size = text.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        ys_in_pad, ys_out_pad = add_sos_eos(text, self.sos, self.eos, self.ignore_id)
        ys_in_lens = text_lengths + 1

        fake_encoder_out = speech.new_zeros(batch_size, 1, self.encoder._output_size)
        # 1. Forward ilm decoder
        ilm_decoder_out, _ = self.decoder.forward_ilm(
            fake_encoder_out, -1, ys_in_pad, ys_in_lens
        )

        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute ilm loss
        loss_ilm = self.criterion_att(ilm_decoder_out, ys_out_pad)

        ilm_acc = th_accuracy(
            ilm_decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        log_softmax_decoder_out = torch.log_softmax(decoder_out, dim=2)
        log_softmax_ilm_decoder_out = torch.log_softmax(ilm_decoder_out, dim=2)

        fused_decoder_out=log_softmax_decoder_out-0.8*log_softmax_ilm_decoder_out
        fused_ilm_acc8 = th_accuracy(
            fused_decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        fused_decoder_out=log_softmax_decoder_out-0.7*log_softmax_ilm_decoder_out
        fused_ilm_acc7 = th_accuracy(
            fused_decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        fused_decoder_out=log_softmax_decoder_out-0.9*log_softmax_ilm_decoder_out
        fused_ilm_acc9 = th_accuracy(
            fused_decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        loss = loss_ilm + 1  # make sure
        stats = dict(
            ilm_loss=loss_ilm.detach(),
            ilm_acc=ilm_acc,
            fused_acc_7=fused_ilm_acc7,
            fused_acc_8=fused_ilm_acc8,
            fused_acc_9=fused_ilm_acc9,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def extract_softlabel(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...) not nessesary it is only used to get device of tensor
            speech_lengths: (Batch, )   not nessesary it is only used to get device of tensor
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
                text.shape[0]
                == text_lengths.shape[0]
        ), (text.shape, text_lengths.shape)
        batch_size = text.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        ys_in_pad, ys_out_pad = add_sos_eos(text, self.sos, self.eos, self.ignore_id)
        ys_in_lens = text_lengths + 1

        fake_encoder_out = speech.new_zeros(batch_size, 1, self.encoder._output_size)
        # 1. Forward ilm decoder
        ilm_decoder_out, _ = self.decoder.forward_ilm(
            fake_encoder_out, -1, ys_in_pad, ys_in_lens
        )

        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )
        log_softmax_decoder_out = torch.log_softmax(decoder_out, dim=2)
        log_softmax_ilm_decoder_out = torch.log_softmax(ilm_decoder_out, dim=2)
        return log_softmax_decoder_out,log_softmax_ilm_decoder_out #the output o


    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase
                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rnnt_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        raise NotImplementedError


class ESPnetASRModel_unadl(ESPnetASRModel):
    def __init__(
            self,
            vocab_size: int,
            token_list: Union[Tuple[str, ...], List[str]],
            frontend: Optional[AbsFrontend],
            specaug: Optional[AbsSpecAug],
            normalize: Optional[AbsNormalize],
            preencoder: Optional[AbsPreEncoder],
            encoder: AbsEncoder,
            postencoder: Optional[AbsPostEncoder],
            decoder: AbsDecoder,
            ctc: CTC,
            rnnt_decoder: None,
            ctc_weight: float = 0.5,
            ignore_id: int = -1,
            lsm_weight: float = 0.0,
            length_normalized_loss: bool = False,
            report_cer: bool = True,
            report_wer: bool = True,
            sym_space: str = "<space>",
            sym_blank: str = "<blank>",
            extract_feats_in_collect_stats: bool = True,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert rnnt_decoder is None, "Not implemented"
        super(ESPnetASRModel, self).__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder
        # we set self.decoder = None in the CTC mode since
        # self.decoder parameters were never used and PyTorch complained
        # and threw an Exception in the multi-GPU experiment.
        # thanks Jeff Farris for pointing out the issue.
        if ctc_weight == 1.0:
            self.decoder = None
        else:
            self.decoder = decoder
        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc
        self.rnnt_decoder = rnnt_decoder
        self.criterion_att = LabelSmoothingLoss_unadl(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.error_calculator = None

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats



    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        forward_ilm=False,
        do_adl=False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        if forward_ilm:
            return self.forward_ilm(speech,speech_lengths,text,text_lengths,do_adl)
        else:
            return self.forward_normal(speech, speech_lengths, text, text_lengths)

    def forward_ilm(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,adl_loss=False,
    ) -> Tuple[torch.Tensor,torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...) not nessesary it is only used to get device of tensor
            speech_lengths: (Batch, )   not nessesary it is only used to get device of tensor
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            text.shape[0]
            == text_lengths.shape[0]
        ), (text.shape, text_lengths.shape)
        batch_size = text.shape[0]
        max_T=text_lengths.max()
        # for data-parallel
        text = text[:, : max_T]

        ys_in_pad, ys_out_pad = add_sos_eos(text, self.sos, self.eos, self.ignore_id)
        ys_in_lens = text_lengths + 1

        fake_encoder_out=speech.new_zeros(batch_size,1,self.encoder._output_size)
        # 1. Forward decoder
        decoder_out, _ = self.decoder.forward_ilm(
            fake_encoder_out, -1, ys_in_pad, ys_in_lens
        )

        # 2. Compute ilm loss
        loss_ilm,unadl_loss = self.criterion_att(decoder_out, ys_out_pad,adl_loss=True)
        ilm_acc = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        if self.rd_aug_text:
            with torch.no_grad():

                aug_mask=torch.rand([self.rd_aug_text_times*batch_size,max_T+1],device=ys_in_pad.device)<=self.rd_aug_text_p
                aug_mask[:,0]=0  #不要修改sos
                rand_ins=torch.randint(0,self.vocab_size-1,[self.rd_aug_text_times*batch_size,max_T+1],device=ys_in_pad.device)
                rand_ins=rand_ins*aug_mask
                rand_ins=rand_ins.type(ys_in_pad.dtype)

                ys_in_pad_rd_aug=torch.remainder(rand_ins+ys_in_pad.repeat(self.rd_aug_text_times, 1), self.vocab_size)#进行文本数据增强，通过按概率替换其中的token实现


            decoder_out_rd_aug, _ = self.decoder.forward_ilm(
                fake_encoder_out, -1, ys_in_pad_rd_aug, ys_in_lens.repeat([self.rd_aug_text_times])
            )
            unadl_loss_aug=self.criterion_att.forward_unadl(decoder_out_rd_aug, ys_out_pad.repeat([self.rd_aug_text_times,1]))


        loss=loss_ilm+1 #make sure

        stats = dict(
            ilm_loss=loss_ilm.detach(),
            unadl_loss=unadl_loss.detach(),
            ilm_acc=ilm_acc,
        )
        if self.rd_aug_text:
            stats['unadl_loss_aug'] = unadl_loss_aug.detach()
            unadl_loss_cp = (unadl_loss + unadl_loss_aug * self.rd_aug_text_times) / (self.rd_aug_text_times + 1)
        else:
            unadl_loss_cp=unadl_loss+1
        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss,unadl_loss_cp, stats, weight = force_gatherable((loss,unadl_loss_cp, stats, batch_size), loss.device)
        return loss,unadl_loss_cp, stats, weight



class ESPnetASRModel_kd(ESPnetASRModel):
    def __init__(
            self,
            vocab_size: int,
            token_list: Union[Tuple[str, ...], List[str]],
            frontend: Optional[AbsFrontend],
            specaug: Optional[AbsSpecAug],
            normalize: Optional[AbsNormalize],
            preencoder: Optional[AbsPreEncoder],
            encoder: AbsEncoder,
            postencoder: Optional[AbsPostEncoder],
            decoder: AbsDecoder,
            ctc: CTC,
            rnnt_decoder: None,
            ctc_weight: float = 0.5,
            ignore_id: int = -1,
            lsm_weight: float = 0.0,
            length_normalized_loss: bool = False,
            report_cer: bool = True,
            report_wer: bool = True,
            sym_space: str = "<space>",
            sym_blank: str = "<blank>",
            extract_feats_in_collect_stats: bool = True,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert rnnt_decoder is None, "Not implemented"
        super(ESPnetASRModel, self).__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder
        # we set self.decoder = None in the CTC mode since
        # self.decoder parameters were never used and PyTorch complained
        # and threw an Exception in the multi-GPU experiment.
        # thanks Jeff Farris for pointing out the issue.
        if ctc_weight == 1.0:
            self.decoder = None
        else:
            self.decoder = decoder
        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc
        self.rnnt_decoder = rnnt_decoder
        self.criterion_att = LabelSmoothingLoss_kd(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.error_calculator = None

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
            self,
            *args, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return super().forward(*args, **kwargs)

    def forward_train(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        ilm_label: torch.Tensor,
        las_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """

        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == ilm_label.shape[0]
        ), (speech.shape, speech_lengths.shape, ilm_label.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        ilm_label = ilm_label[:, : text_lengths.max()+1]
        las_label = las_label[:, : text_lengths.max()+1]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # 2a. Attention-decoder branch
        if self.ctc_weight == 1.0:
            loss_att_org,loss_att_kd, acc_att, cer_att, wer_att = None, None, None, None
        else:
            loss_att_org,loss_att_kd, acc_att, cer_att, wer_att = self._calc_att_loss_kd(
                encoder_out, encoder_out_lens, text,text_lengths,ilm_label,las_label
            )

        # 2b. CTC branch
        if self.ctc_weight == 0.0:

            loss_ctc, cer_ctc = None, None
        else:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        # 2c. RNN-T branch
        if self.rnnt_decoder is not None:
            _ = self._calc_rnnt_loss(encoder_out, encoder_out_lens, text, text_lengths)
        loss_att = (1 - self.kd_alpha) * loss_att_org + self.kd_alpha * loss_att_kd

        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att_org.detach() if loss_att is not None else None,
            loss_att_kd=loss_att_kd.detach() if loss_att_kd is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            acc=acc_att,
            cer=cer_att,
            wer=wer_att,
            cer_ctc=cer_ctc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight



    def _calc_att_loss_kd(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens : torch.Tensor,
        ilm_label: torch.Tensor,
        las_label: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        with torch.no_grad():
            teacher_labels=las_label-self.kd_ilme_factor*ilm_label
            teacher_labels=torch.nn.functional.softmax(teacher_labels / self.kd_T,dim=2)
        # 2. Compute attention loss
        loss_att,kd_loss = self.criterion_att(decoder_out, ys_out_pad,teacher_labels=teacher_labels)

        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att,kd_loss, acc_att, cer_att, wer_att


    def kd_loss(self,outputs,labels,teacher_outputs):



        KD_loss = torch.nn.KLDivLoss()(torch.nn.functional.log_softmax(outputs / self.kd_T, dim=1),
                                 torch.nn.functional.softmax(teacher_outputs / self.kd_T, dim=1)) * ( self.kd_alpha * self.kd_T * self.kd_T) + \
                  torch.nn.functional.cross_entropy(outputs, labels) * (1. -  self.kd_alpha)
        return KD_loss


class ESPnetASRModel_unaug(ESPnetASRModel):

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # 2a. Attention-decoder branch
        if self.ctc_weight == 1.0:
            loss_att, acc_att, cer_att, wer_att = None, None, None, None
        else:
            loss_att, acc_att, cer_att, wer_att,unaug_loss = self._calc_att_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        # 2b. CTC branch
        if self.ctc_weight == 0.0:
            loss_ctc, cer_ctc = None, None
        else:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        # 2c. RNN-T branch
        if self.rnnt_decoder is not None:
            _ = self._calc_rnnt_loss(encoder_out, encoder_out_lens, text, text_lengths)

        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
        loss+=self.unaug_conf["unaug_loss_weight"]*unaug_loss

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            unaug_loss=unaug_loss.detach(),
            acc=acc_att,
            cer=cer_att,
            wer=wer_att,
            cer_ctc=cer_ctc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())


        # 3. Compute unaug loss
        #3.1 生成替换后的ys
        with torch.no_grad():
            replaced_ys_in_pad,unaug_mask=self.replace_ys(ys_pad,ys_pad_lens-1,decoder_out)
            #unaug_mask True的地方是需要使用un kl训练的，






        # 3.2 forward decoder with ys with errors
        unaug_decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, replaced_ys_in_pad, ys_in_lens
        )

        #3.3 cal unaug_loss
        unaug_loss=self.__cal_unaug_loss(unaug_decoder_out,unaug_mask)


        return loss_att, acc_att, cer_att, wer_att,unaug_loss


    def __cal_unaug_loss(self,unaug_decoder_out,mask):
        with torch.no_grad():
            un_dist = unaug_decoder_out.clone()
            un_dist.fill_(1 / self.vocab_size)  #生成均匀分布


        batch_size = unaug_decoder_out.size(0)
        T = unaug_decoder_out.size(1)
        unaug_decoder_out = unaug_decoder_out.view(-1, self.vocab_size)
        unaug_loss_all=torch.nn.functional.kl_div(torch.log_softmax(unaug_decoder_out, dim=1), un_dist.view(-1,self.vocab_size), reduction="none")
        total=mask.sum()

        return unaug_loss_all.masked_fill(~mask.view(batch_size*T,-1), 0).sum() / total

    def replace_ys(self,ys_pad,ys_pad_lens,decoder_out):
        #注意 ys_pad_lens 应该是原始长度
        # 通过将ys替换成decoder_out中预测概率较高的前几个符号，来生成假的ys。保留ground_truth的概率是replace_p[0]
                    #替换为decoder_out中预测概率第二高的概率是replace_p[1]（训练快结束的时候decoder_out预测概率最高的通常就是ground_truth）
                    #替换为decoder_out中预测概率第三高的概率是replace_p[2]
                    #值得注意的是有时候ground_truth会在decoder_out 第二或者第三的位置，此时不进行增强，因此实际的替换概率会略低于replace_p[0]。

        B = ys_pad.size(0)
        max_source_length = ys_pad.size(1)

        replace_P = self.unaug_conf["replace_p"]

        decoder_out_sort = decoder_out.argsort(dim=-1, descending=True)  # 返回的是index

        m = Categorical(torch.tensor(replace_P))
        gg = m.sample((B, max_source_length))
        gg=gg.to(decoder_out_sort.device)
        replaced_ys_pad = torch.gather(decoder_out_sort, dim=-1, index=gg.view(B, max_source_length, 1)).view(B,
                                                                                                              max_source_length) #替换为decoder_out中概率最大的前几
        replaced_ys_pad = torch.where(gg != 0, replaced_ys_pad.view(B, max_source_length), ys_pad)  # gg为0的时候不替换
        augmentd_mask = replaced_ys_pad != ys_pad  # 虽然进行了替换，但是依然有替换后与ground_truth相同的可能。因此与ground_truth不同的才是真的增强过的
        # augmentd_mask=gg!=0

        # print("replace rate",augmentd_mask.sum()/(B* max_source_length))
        # for i in range(B):
        #     for j in range(max_source_length):
        #         if replaced_ys_pad[i][j]!=ys_pad[i][j]:
        #             assert replaced_ys_pad[i][j]==decoder_out_sort[i][j][gg[i][j]]
        #             assert augmentd_mask[i][j]==True
        #         else:
        #             assert augmentd_mask[i][j] == False
        #         # 验证代码
        replaced_ys_pad_in,_=add_sos_eos(replaced_ys_pad, self.sos, self.eos, self.ignore_id)



        no_replace_length=torch.argmax(augmentd_mask.int(), dim=1)  # 没有被替换的长度argmax 只会返回最靠近开始的最大值，利用这个特性可以计算出出现替换前bpe序列的长度

        unaug_sample = torch.sum(augmentd_mask, dim=1).bool()  #没有发生替换的样本


        ids = torch.arange(0, max_source_length, device=ys_pad_lens.device)
        mask1 = (ids <= ys_pad_lens.unsqueeze(1)).bool()
        mask2 = (ids >= no_replace_length.unsqueeze(1)).bool()
        mask = mask1 * mask2 * unaug_sample.view(-1, 1)
        mask=torch.nn.functional.pad(input=mask, pad=(1,0), mode='constant', value=False)

        mask.to(decoder_out.device)
        replaced_ys_pad_in = replaced_ys_pad_in.to(ys_pad.device)

        return replaced_ys_pad_in,mask  #替换后的ys，已经计算loss时使用的mask，只有mask为True的地方需要计算un loss