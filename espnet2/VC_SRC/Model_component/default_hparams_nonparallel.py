
class HParams(object):

    def __init__(self,**default_param):

        temp_local=self.__dict__


        for key in default_param.keys():
            temp_local[key]=default_param[key]


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = HParams(
        total_iteration=150000,



        ################################
        # BN_feat Parameters        #
        ################################
        BN_feat_dim=512,
        num_of_spk=24,
        n_mel_channels=80,
        speaker_emb_dim=512,

        ################################
        # Model Parameters             #
        ################################
        prenet_depths=[ 512, 256 ], #这是decoder里面的prenent
        encoder_embedding_dim=-1, #必须等于prenet_dim，在下面赋值


        # Decoder parameters
        n_frames_per_step=2,
        decoder_rnn_dim=512,
        prenet_dim=512,
        max_decoder_steps=200000,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=512,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batchsize=32,

    )

    hparams.encoder_embedding_dim=hparams.prenet_depths[-1]

    return hparams
