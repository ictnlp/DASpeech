import logging
import torch
import random
import numpy as np
import torch.nn as nn

from fairseq import checkpoint_utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import Embedding
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.data.audio.speech_to_text_dataset import _collate_frames

from DASpeech.models.fastspeech2_noemb import FastSpeech2EncoderNoEmb
from DASpeech.generator.s2s_nat_generator import NATS2SDecoderOut
from DASpeech.models.s2t_conformer_dag import (
    torch_seed,
    S2TConformerDAGModel,
)

logger = logging.getLogger(__name__)


class FFNAdapter(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(FFNAdapter, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


@register_model("s2s_conformer_dag_fastspeech2")
class S2SConformerDAGFastSpeech2Model(S2TConformerDAGModel):

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        args.tgt_dict_size = len(task.target_dictionary)
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)

        # DA-Transformer
        base_model = cls(args, encoder, decoder)
        if getattr(args, "load_pretrained_dag_from", None):
            state_dict = checkpoint_utils.load_checkpoint_to_cpu(args.load_pretrained_dag_from)["model"]
            base_model.load_state_dict(state_dict)
            logger.info(f"Successfully load pretrained DA-Transformer from {args.load_pretrained_dag_from}.")
        # Adaptor
        base_model.adaptor = FFNAdapter(
            args.decoder_embed_dim,
            args.adaptor_ffn_dim,
            args.tts_encoder_embed_dim,
            args.dropout,
        )
        # FastSpeech2
        base_model.tts = FastSpeech2EncoderNoEmb(args, task.target_dictionary, None)
        if getattr(args, "load_pretrained_fastspeech_from", None):
            base_model.tts = checkpoint_utils.load_pretrained_component_from_model(
                component=base_model.tts, checkpoint=args.load_pretrained_fastspeech_from,
            )
            logger.info(f"Successfully load pretrained FastSpeech2 from {args.load_pretrained_fastspeech_from}.")

        return base_model

    @staticmethod
    def add_args(parser):
        S2TConformerDAGModel.add_args(parser)
        parser.add_argument(
            "--load-pretrained-fastspeech-from",
            type=str,
            help="path to pretrained fastspeech2 model",
        )
        # Adaptor
        parser.add_argument("--adaptor-ffn-dim", type=int)
        # fastspeech2 args
        parser.add_argument("--output-frame-dim", type=int)
        parser.add_argument("--speaker-embed-dim", type=int)
        # FFT blocks
        parser.add_argument("--fft-hidden-dim", type=int)
        parser.add_argument("--fft-kernel-size", type=int)
        parser.add_argument("--tts-encoder-layers", type=int)
        parser.add_argument("--tts-encoder-embed-dim", type=int)
        parser.add_argument("--tts-encoder-attention-heads", type=int)
        parser.add_argument("--tts-decoder-layers", type=int)
        parser.add_argument("--tts-decoder-embed-dim", type=int)
        parser.add_argument("--tts-decoder-attention-heads", type=int)
        # variance predictor
        parser.add_argument("--var-pred-n-bins", type=int)
        parser.add_argument("--var-pred-hidden-dim", type=int)
        parser.add_argument("--var-pred-kernel-size", type=int)
        parser.add_argument("--var-pred-dropout", type=float)
        # postnet
        parser.add_argument("--add-postnet", action="store_true")
        parser.add_argument("--postnet-dropout", type=float)
        parser.add_argument("--postnet-layers", type=int)
        parser.add_argument("--postnet-conv-dim", type=int)
        parser.add_argument("--postnet-conv-kernel-size", type=int)
    
    def extract_features(self, prev_output_tokens, encoder_out, rand_seed, require_links=False):
        with torch_seed(rand_seed):
            features, _ = self.decoder.extract_features(
                prev_output_tokens,
                encoder_out=encoder_out,
                embedding_copy=False
            )
            # word_ins_out = self.decoder.output_layer(features)
            word_ins_out = self.decoder.output_projection(features)

            links = None
            if require_links:
                links = self.extract_links(features, \
                            prev_output_tokens, \
                            self.decoder.link_positional, \
                            self.decoder.query_linear, \
                            self.decoder.key_linear, \
                            self.decoder.gate_linear
                        )

        return word_ins_out, links, features

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat=None, glat_function=None, **kwargs
    ):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        rand_seed = random.randint(0, 19260817)
        # decoding
        glat_info = None
        if glat and tgt_tokens is not None:
            with torch.set_grad_enabled(glat.get('require_glance_grad', False)):
                word_ins_out, links, _ = self.extract_features(prev_output_tokens, encoder_out, rand_seed, require_links=True)
                prev_output_tokens, tgt_tokens, glat_info = glat_function(self, word_ins_out, tgt_tokens, prev_output_tokens, glat, links=links)
                word_ins_out = None

        word_ins_out, links, features = self.extract_features(prev_output_tokens, encoder_out, rand_seed, require_links=True)

        ret = {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "nll_loss": True,
                "features": features,
            }
        }
        ret['links'] = links

        if glat_info is not None:
            ret.update(glat_info)
        return ret
    
    def initialize_output_tokens(self, encoder_out, src_tokens, src_lengths):
        length_tgt = (src_lengths * self.args.src_upsample_scale).long().clamp_(min=2, max=self.decoder.max_positions())
        initial_output_tokens = self.initialize_output_tokens_with_length(src_tokens, length_tgt)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return NATS2SDecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            features=None,  # add "features" in the dict
            features_padding_mask=None,  # add "features_padding_mask" in the dict
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens

        history = decoder_out.history
        rand_seed = random.randint(0, 19260817)

        # execute the decoder
        output_logits, links, features = self.extract_features(output_tokens, encoder_out, rand_seed, require_links=True)
        if self.args.max_transition_length != -1:
            links = self.restore_valid_links(links)
        output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1)

        output_logits_normalized = output_logits.log_softmax(dim=-1)
        unreduced_logits, unreduced_tokens = output_logits_normalized.max(dim=-1)
        unreduced_tokens = unreduced_tokens.tolist()  # y_i = argmax_y P(y | v_i)

        if self.args.decode_strategy in ["lookahead", "greedy"]:
            if self.args.decode_strategy == "lookahead":
                output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1).tolist()
                links_idx = (links + unreduced_logits.unsqueeze(1) * self.args.decode_beta).max(dim=-1)[1].cpu().tolist() # batch * prelen
            elif self.args.decode_strategy == "greedy":
                output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1).tolist()
                links_idx = links.max(dim=-1)[1].cpu().tolist() # batch * prelen

            unpad_output_tokens = []
            unpad_output_features = []
            for i, length in enumerate(output_length):
                last = unreduced_tokens[i][0]  # <bos>
                j = 0
                res = [last]
                res_features = []  # not include <bos> hidden states
                while j != length - 1:
                    j = links_idx[i][j]  # transition
                    now_token = unreduced_tokens[i][j]  # emission
                    # NOTE: the original DA-Transformer remove same consecutive tokens
                    if now_token != self.tgt_dict.pad_index and now_token != last:
                        res.append(now_token)
                        res_features.append(features[i][j].unsqueeze(0))
                    last = now_token
                res_features = torch.cat(res_features, dim=0)
                unpad_output_tokens.append(res)
                unpad_output_features.append(res_features)

            output_seqlen = max([len(res) for res in unpad_output_tokens])
            output_tokens = [res + [self.tgt_dict.pad_index] * (output_seqlen - len(res)) for res in unpad_output_tokens]
            output_tokens = torch.tensor(output_tokens, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
            output_features = _collate_frames(unpad_output_features)
            output_features_length = torch.tensor([len(res_features) for res_features in unpad_output_features])
            output_features_padding_mask = lengths_to_padding_mask(output_features_length)
        elif self.args.decode_strategy in ["viterbi", "jointviterbi"]:
            scores = []
            indexs = []
            # batch * graph_length
            alpha_t = links[:,0]
            if self.args.decode_strategy == "jointviterbi":
                alpha_t += unreduced_logits[:,0].unsqueeze(1) * self.args.decode_beta
            batch_size, graph_length, _ = links.size()
            alpha_t += unreduced_logits * self.args.decode_beta
            scores.append(alpha_t)
            
            # the exact max_length should be graph_length - 2, but we can reduce it to an appropriate extent to speedup decoding
            max_length = int(graph_length / 8 / self.args.src_upsample_scale)
            for i in range(max_length - 1):
                alpha_t, index = torch.max(alpha_t.unsqueeze(-1) + links, dim = 1)
                if self.args.decode_strategy == "jointviterbi":
                    alpha_t += unreduced_logits * self.args.decode_beta
                scores.append(alpha_t)
                indexs.append(index)

            # max_length * batch * graph_length
            indexs = torch.stack(indexs, dim = 0)
            scores = torch.stack(scores, dim = 0)
            link_last = torch.gather(links, -1, (output_length - 1).view(batch_size, 1, 1).repeat(1, graph_length, 1)).view(1, batch_size, graph_length)
            scores += link_last

            # max_length * batch
            scores, max_idx = torch.max(scores, dim = -1)
            lengths = torch.arange(max_length).unsqueeze(-1).repeat(1, batch_size) + 1
            length_penalty = (lengths ** self.args.decode_viterbibeta).cuda(scores.get_device())
            scores = scores / length_penalty
            max_score, pred_length = torch.max(scores, dim = 0)
            pred_length = pred_length + 1

            initial_idx = torch.gather(max_idx, 0, (pred_length - 1).view(1, batch_size)).view(batch_size).tolist()
            unpad_output_tokens = []
            unpad_output_features = []
            indexs = indexs.tolist()
            pred_length = pred_length.tolist()
            for i, length in enumerate(pred_length):
                j = initial_idx[i]
                last = unreduced_tokens[i][j]
                res = [last]
                res_features = [features[i][j].unsqueeze(0)]
                for k in range(length - 1):
                    j = indexs[length - k - 2][i][j]
                    now_token = unreduced_tokens[i][j]
                    if now_token != self.tgt_dict.pad_index and now_token != last:
                        res.insert(0, now_token)
                        res_features.insert(0, features[i][j].unsqueeze(0))
                    last = now_token
                res_features = torch.cat(res_features, dim=0)  # not include <bos> hidden states
                unpad_output_tokens.append(res)
                unpad_output_features.append(res_features)

            output_seqlen = max([len(res) for res in unpad_output_tokens])
            output_tokens = [res + [self.tgt_dict.pad_index] * (output_seqlen - len(res)) for res in unpad_output_tokens]
            output_tokens = torch.tensor(output_tokens, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
            output_features = _collate_frames(unpad_output_features)
            output_features_length = torch.tensor([len(res_features) for res_features in unpad_output_features])
            output_features_padding_mask = lengths_to_padding_mask(output_features_length)
        elif self.args.decode_strategy == "beamsearch":

            batch_size, prelen, _ = links.shape

            assert batch_size <= self.args.decode_max_batchsize, "Please set --decode-max-batchsize for beamsearch with a larger batch size"

            top_logits, top_logits_idx = output_logits.log_softmax(dim=-1).topk(self.args.decode_top_cand_n, dim=-1)
            dagscores_arr = (links.unsqueeze(-1) + top_logits.unsqueeze(1) * self.args.decode_beta)  # batch * prelen * prelen * top_cand_n
            dagscores, top_cand_idx = dagscores_arr.reshape(batch_size, prelen, -1).topk(self.args.decode_top_cand_n, dim=-1) # batch * prelen * top_cand_n

            nextstep_idx = torch.div(top_cand_idx, self.args.decode_top_cand_n, rounding_mode="floor") # batch * prelen * top_cand_n
            logits_idx_idx = top_cand_idx % self.args.decode_top_cand_n # batch * prelen * top_cand_n
            idx1 = torch.arange(batch_size, device=links.device).unsqueeze(-1).unsqueeze(-1).expand(*nextstep_idx.shape)
            logits_idx = top_logits_idx[idx1, nextstep_idx, logits_idx_idx] # batch * prelen * top_cand_n

            rearange_idx = logits_idx.sort(dim=-1)[1]
            dagscores = dagscores.gather(-1, rearange_idx) # batch * prelen * top_cand_n
            nextstep_idx = nextstep_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n
            logits_idx = logits_idx.gather(-1, rearange_idx) # batch * prelen * top_cand_n

            dagscores = np.ascontiguousarray(dagscores.cpu().numpy())
            nextstep_idx = np.ascontiguousarray(nextstep_idx.int().cpu().numpy())
            logits_idx = np.ascontiguousarray(logits_idx.int().cpu().numpy())
            output_length_cpu = np.ascontiguousarray(output_length.int().cpu().numpy())

            res, score = self.dag_search.dag_search(dagscores, nextstep_idx, logits_idx,
                output_length_cpu,
                self.args.decode_alpha,
                self.args.decode_gamma,
                self.args.decode_beamsize,
                self.args.decode_max_beam_per_length,
                self.args.decode_top_p,
                self.tgt_dict.pad_index,
                self.tgt_dict.bos_index,
                1 if self.args.decode_dedup else 0
            )
            output_tokens = torch.tensor(res, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
            output_scores = torch.tensor(score, device=decoder_out.output_scores.device, dtype=decoder_out.output_scores.dtype).unsqueeze(dim=-1).expand(*output_tokens.shape)

        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=torch.full(output_tokens.size(), 1.0),
            features=output_features,
            features_padding_mask=output_features_padding_mask,
            attn=None,
            history=history,
        )


@register_model_architecture(
    "s2s_conformer_dag_fastspeech2", "s2s_conformer_dag_fastspeech2"
)
def base_architecture(args):
    # conformer args
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    args.attn_type = getattr(args, "attn_type", None)
    args.pos_enc_type = getattr(args, "pos_enc_type", "abs")
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.input_channels = getattr(args, "input_channels", 1)
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")  # for Conv1d
    args.conv_channels = getattr(args, "conv_channels", 1024)  # for Conv1d
    args.conv_out_channels = getattr(args, "conv_out_channels", 256)  # for Conv2d
    args.conv_version = getattr(args, "conv_version", "s2t_transformer")
    args.max_source_positions = getattr(args, "max_source_positions", 6000)
    args.depthwise_conv_kernel_size = getattr(args, "depthwise_conv_kernel_size", 31)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    
    # dag args
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)

    # adaptor args
    args.adaptor_ffn_dim = getattr(args, "adaptor_ffn_dim", 1024)
    # fastspeech2 args
    # NOTE: use same dropout and attention_dropout as DA-Transformer
    args.output_frame_dim = getattr(args, "output_frame_dim", 80)
    args.speaker_embed_dim = getattr(args, "speaker_embed_dim", 64)
    # FFT blocks
    args.fft_hidden_dim = getattr(args, "fft_hidden_dim", 1024)
    args.fft_kernel_size = getattr(args, "fft_kernel_size", 9)
    args.tts_encoder_layers = getattr(args, "tts_encoder_layers", 4)
    args.tts_encoder_embed_dim = getattr(args, "tts_encoder_embed_dim", 256)
    args.tts_encoder_attention_heads = getattr(args, "tts_encoder_attention_heads", 2)
    args.tts_decoder_layers = getattr(args, "tts_decoder_layers", 4)
    args.tts_decoder_embed_dim = getattr(args, "tts_decoder_embed_dim", 256)
    args.tts_decoder_attention_heads = getattr(args, "tts_decoder_attention_heads", 2)
    # variance predictor
    args.var_pred_n_bins = getattr(args, "var_pred_n_bins", 256)
    args.var_pred_hidden_dim = getattr(args, "var_pred_hidden_dim", 256)
    args.var_pred_kernel_size = getattr(args, "var_pred_kernel_size", 3)
    args.var_pred_dropout = getattr(args, "var_pred_dropout", 0.5)
    # postnet
    args.add_postnet = getattr(args, "add_postnet", False)
    args.postnet_dropout = getattr(args, "postnet_dropout", 0.5)
    args.postnet_layers = getattr(args, "postnet_layers", 5)
    args.postnet_conv_dim = getattr(args, "postnet_conv_dim", 512)
    args.postnet_conv_kernel_size = getattr(args, "postnet_conv_kernel_size", 5)
