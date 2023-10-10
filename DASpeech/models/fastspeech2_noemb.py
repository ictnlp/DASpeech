import torch
import torch.nn as nn

from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
)
from fairseq.models.text_to_speech.fastspeech2 import (
    model_init,
    Embedding,
    FastSpeech2Encoder,
    FFTLayer,
    VarianceAdaptor,
    VariancePredictor,
    LengthRegulator,
    Postnet,
)
from fairseq.modules import (
    FairseqDropout,
    PositionalEmbedding,
)


class VariancePredictorNoEmb(VariancePredictor):
    def __init__(self, args):
        nn.Module.__init__(self)
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                args.tts_encoder_embed_dim,
                args.var_pred_hidden_dim,
                kernel_size=args.var_pred_kernel_size,
                padding=(args.var_pred_kernel_size - 1) // 2,
            ),
            nn.ReLU(),
        )
        self.ln1 = nn.LayerNorm(args.var_pred_hidden_dim)
        self.dropout_module = FairseqDropout(
            p=args.var_pred_dropout, module_name=self.__class__.__name__
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                args.var_pred_hidden_dim,
                args.var_pred_hidden_dim,
                kernel_size=args.var_pred_kernel_size,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.ln2 = nn.LayerNorm(args.var_pred_hidden_dim)
        self.proj = nn.Linear(args.var_pred_hidden_dim, 1)


class VarianceAdaptorNoEmb(VarianceAdaptor):
    def __init__(self, args):
        nn.Module.__init__(self)
        self.args = args
        self.length_regulator = LengthRegulator()
        self.duration_predictor = VariancePredictorNoEmb(args)
        self.pitch_predictor = VariancePredictorNoEmb(args)
        self.energy_predictor = VariancePredictorNoEmb(args)

        n_bins, steps = self.args.var_pred_n_bins, self.args.var_pred_n_bins - 1
        self.pitch_bins = torch.linspace(args.pitch_min, args.pitch_max, steps)
        self.embed_pitch = Embedding(n_bins, args.tts_encoder_embed_dim)
        self.energy_bins = torch.linspace(args.energy_min, args.energy_max, steps)
        self.embed_energy = Embedding(n_bins, args.tts_encoder_embed_dim)


class FastSpeech2EncoderNoEmb(FastSpeech2Encoder):
    """
        Modified FastSpeech2 which accepts hidden states (x) as input rather than src_tokens.
    """
    def __init__(self, args, src_dict, embed_speaker):
        FairseqEncoder.__init__(self, src_dict)
        self.args = args
        self.padding_idx = src_dict.pad()
        self.n_frames_per_step = args.n_frames_per_step
        self.out_dim = args.output_frame_dim * args.n_frames_per_step

        self.embed_speaker = embed_speaker
        self.spk_emb_proj = None
        if embed_speaker is not None:
            self.spk_emb_proj = nn.Linear(
                args.tts_encoder_embed_dim + args.speaker_embed_dim, args.tts_encoder_embed_dim
            )

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_tokens = Embedding(
            len(src_dict), args.tts_encoder_embed_dim, padding_idx=self.padding_idx
        )

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, args.tts_encoder_embed_dim, self.padding_idx
        )
        self.pos_emb_alpha = nn.Parameter(torch.ones(1))
        self.dec_pos_emb_alpha = nn.Parameter(torch.ones(1))

        self.encoder_fft_layers = nn.ModuleList(
            FFTLayer(
                args.tts_encoder_embed_dim,
                args.tts_encoder_attention_heads,
                args.fft_hidden_dim,
                args.fft_kernel_size,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
            )
            for _ in range(args.tts_encoder_layers)
        )

        self.var_adaptor = VarianceAdaptorNoEmb(args)

        self.decoder_fft_layers = nn.ModuleList(
            FFTLayer(
                args.tts_decoder_embed_dim,
                args.tts_decoder_attention_heads,
                args.fft_hidden_dim,
                args.fft_kernel_size,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
            )
            for _ in range(args.tts_decoder_layers)
        )

        self.out_proj = nn.Linear(args.tts_decoder_embed_dim, self.out_dim)

        self.postnet = None
        if args.add_postnet:
            self.postnet = Postnet(
                self.out_dim,
                args.postnet_conv_dim,
                args.postnet_conv_kernel_size,
                args.postnet_layers,
                args.postnet_dropout,
            )

        self.apply(model_init)
    
    def forward(
        self,
        x,
        enc_padding_mask,
        speaker=None,
        durations=None,
        pitches=None,
        energies=None,
        **kwargs
    ):
        x += self.pos_emb_alpha * self.embed_positions(enc_padding_mask)
        x = self.dropout_module(x)

        for layer in self.encoder_fft_layers:
            x = layer(x, enc_padding_mask)

        if self.embed_speaker is not None:
            bsz, seq_len, _ = x.size()
            emb = self.embed_speaker(speaker).expand(bsz, seq_len, -1)
            x = self.spk_emb_proj(torch.cat([x, emb], dim=2))

        x, out_lens, log_dur_out, pitch_out, energy_out = self.var_adaptor(
            x, enc_padding_mask, durations, pitches, energies
        )

        dec_padding_mask = lengths_to_padding_mask(out_lens)
        x += self.dec_pos_emb_alpha * self.embed_positions(dec_padding_mask)
        for layer in self.decoder_fft_layers:
            x = layer(x, dec_padding_mask)

        x = self.out_proj(x)
        x_post = None
        if self.postnet is not None:
            x_post = x + self.postnet(x)
        return x, x_post, out_lens, log_dur_out, pitch_out, energy_out