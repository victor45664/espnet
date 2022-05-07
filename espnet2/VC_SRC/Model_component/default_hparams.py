
class HParams(object):

    def __init__(self,**default_param):

        temp_local=self.__dict__


        for key in default_param.keys():
            temp_local[key]=default_param[key]


def create_hparams():
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = HParams(
        total_iteration=150000,


        source_feat_dim=80,  #输入语音特征的维度
        n_mel_channels=80,    #输出mel谱的维度


        ################################
        # Model Parameters             #
        ################################

        encoder_embedding_dim=-1, #必须等于prenet_dim，在下面赋值

        # Encoder parameters
        encoder_conf={
            'attention_heads': 4,
            'linear_units': 256,
            'num_blocks': 6,
            'positional_dropout_rate': 0.1,
            'attention_dropout_rate': 0.1,
            'input_layer': 'conv2d',
            'positionwise_layer_type': 'linear',
            'positionwise_conv_kernel_size': 3,
            'rel_pos_type': 'latest',
            'pos_enc_layer_type': 'rel_pos',
            'selfattention_layer_type': 'rel_selfattn',
            'activation_type': 'swish',
            'zero_triu': False,
            'cnn_module_kernel': 31,
            'padding_idx': -1,
            'concat_after': False,
            'dropout_rate': 0.1,
            'macaron_style': True,
            'normalize_before': True,
            'use_cnn_module': True,
            "output_size": 512
        },


        # Decoder parameters
        n_frames_per_step=1,
        decoder_rnn_dim=512,
        prenet_dim=512,
        max_decoder_steps=200000,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,
        gate_threshold=0.5,


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
        mask_padding=True  # set model's padded outputs to padded values
    )

    hparams.encoder_embedding_dim=hparams.encoder_conf["output_size"]

    return hparams
