import sys
import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.data.audio.speech_to_text_dataset import _collate_frames
from fairseq.data.data_utils import lengths_to_padding_mask, lengths_to_mask
from DASpeech.criterions.nat_dag_loss import (
    NATDAGLoss,
    dag_logsoftmax_gather_inplace,
    torch_dag_logsoftmax_gather_inplace,
    dag_best_alignment,
    torch_dag_best_alignment,
)
from ..custom_ops import dag_loss_with_alpha_beta, logsumexp_keepdim

########### gpu use tracker ###########
# import inspect
SHOW_MEMORY_USE=False
if SHOW_MEMORY_USE:
    from fairseq.gpu_mem_track import MemTracker
    gpu_tracker = MemTracker()
########################################

@register_criterion("s2s_dag_fastspeech2_loss")
class S2SDAGFastSpeech2Loss(NATDAGLoss):

    @staticmethod
    def add_args(parser):
        NATDAGLoss.add_args(parser)
        parser.add_argument(
            "--training-strategy",
            type=str,
            choices=["expect", "argmax"],
            help="training strategy: \
                  expect: z_i = \sum_j P(a_i = j | x, y) * v_j \
                  argmax: z_i = v_{a*_i}, a* = argmax P(y, a | x)"
        )
        parser.add_argument(
            "--tts-loss-weight",
            type=float,
            default=1.0,
            help="weight of tts loss"
        )
        parser.add_argument(
            "--dag-freezing-steps",
            type=int,
            default=-1,
            help="freezing steps of DA-Transformer"
        )

    def _compute_dag_loss_with_alpha_beta(self, outputs, output_masks, targets, target_masks, links, label_smoothing=0.0, name="loss",
                factor=1.0, matchmask=None, keep_word_mask=None, model=None):

        batch_size = outputs.shape[0]
        prelen = outputs.shape[1]
        tarlen = targets.shape[1]

        output_length = output_masks.sum(dim=-1)
        target_length = target_masks.sum(dim=-1)

        if self.cfg.torch_dag_logsoftmax_gather:
            outputs, match_all = torch_dag_logsoftmax_gather_inplace(outputs, targets.unsqueeze(1).expand(-1, prelen, -1))
        else:
            outputs, match_all = dag_logsoftmax_gather_inplace(outputs, targets.unsqueeze(1).expand(-1, prelen, -1))
        match_all = match_all.transpose(1, 2)

        if matchmask is not None and not self.cfg.no_force_emit:
            glat_prev_mask = keep_word_mask.unsqueeze(1)
            match_all = match_all.masked_fill(glat_prev_mask, 0) + match_all.masked_fill(~matchmask, float("-inf")).masked_fill(~glat_prev_mask, 0).detach()
        nvalidtokens = output_masks.sum()

        assert not self.cfg.torch_dag_loss, "must use cuda version loss to obtain alpha and beta"
        assert model.args.max_transition_length != -1, "cuda dag loss does not support max_transition_length=-1. You can use a very large number such as 99999"
        loss_result, (alpha, beta) = dag_loss_with_alpha_beta(match_all, links, output_length, target_length)

        invalid_masks = loss_result.isinf().logical_or(loss_result.isnan())
        loss_result.masked_fill_(invalid_masks, 0)
        invalid_nsentences = invalid_masks.sum().detach()

        loss = -(loss_result / target_length).mean()
        nll_loss = loss.detach()
        nsentences, ntokens = targets.shape[0], targets.ne(self.task.tgt_dict.pad()).sum()

        loss_nofactor = loss
        loss = loss * factor

        return {"name": name, "loss": loss, "nll_loss": nll_loss,
                "factor": factor, "ntokens": ntokens, "nvalidtokens": nvalidtokens, "nsentences": nsentences,
                "loss_nofactor": loss_nofactor, "invalid_nsentences": invalid_nsentences}, alpha, beta

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # import gc
        # gc.collect()
        if SHOW_MEMORY_USE:
            print(torch.cuda.memory_reserved() / 1024 / 1024, file=sys.stderr, flush=True)
            gpu_tracker.clear_cache()
        # gpu_tracker.track()

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens = sample["target_text"]

        if SHOW_MEMORY_USE:
            print(sample["net_input"]["src_tokens"].shape[0], sample["net_input"]["src_tokens"].shape[1], tgt_tokens.shape[1], file=sys.stderr, end=" ")

        if sample.get("update_num", None) is not None: # in training
            self.set_update_num(sample['update_num'])

        prev_output_tokens = model.initialize_output_tokens_by_tokens(src_tokens, src_lengths)

        if self.glat_p == 0:
            glat = None
        else:
            glat = {
                "context_p": max(self.glat_p, 0),
                "require_glance_grad": False
            }

        def glat_function(model, word_ins_out, tgt_tokens, prev_output_tokens, glat, links=None):
            batch_size, prelen, _ = links.shape
            tarlen = tgt_tokens.shape[1]
            nonpad_positions = ~tgt_tokens.eq(model.pad)
            target_length = (nonpad_positions).sum(1)
            output_length = prev_output_tokens.ne(model.pad).sum(1)

            pred_tokens = word_ins_out.argmax(-1)
            if self.cfg.torch_dag_logsoftmax_gather:
                word_ins_out, match = torch_dag_logsoftmax_gather_inplace(word_ins_out, tgt_tokens.unsqueeze(1).expand(-1, prelen, -1))
            else:
                word_ins_out, match = dag_logsoftmax_gather_inplace(word_ins_out, tgt_tokens.unsqueeze(1).expand(-1, prelen, -1))
            match = match.transpose(1, 2)

            if self.cfg.torch_dag_best_alignment:
                if model.args.max_transition_length != -1:
                    links = model.restore_valid_links(links)
                path = torch_dag_best_alignment(match, links, output_length, target_length)
            else:
                assert model.args.max_transition_length != -1, "cuda dag best alignment does not support max_transition_length=-1. You can use a very large number such as 99999"
                path = dag_best_alignment(match, links, output_length, target_length) # batch * prelen

            predict_align_mask = path >= 0
            matchmask = torch.zeros(batch_size, tarlen + 1, prelen, device=match.device, dtype=torch.bool).scatter_(1, path.unsqueeze(1) + 1, 1)[:, 1:]
            oracle = tgt_tokens.gather(-1, path.clip(min=0)) # bsz * prelen
            same_num = ((pred_tokens == oracle) & predict_align_mask).sum(1)

            if self.glance_strategy is None:
                keep_prob = ((target_length - same_num) / target_length * glat['context_p']).unsqueeze(-1) * predict_align_mask.float()

            elif self.glance_strategy in ['number-random']:
                prob = torch.randn(oracle.shape, device=tgt_tokens.device, dtype=torch.float)
                prob.masked_fill_(~predict_align_mask, -100)
                glance_nums = ((target_length - same_num) * glat['context_p'] + 0.5).to(torch.long)
                #prob_thresh = prob.topk(glance_nums.max().clip(min=1))[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
                prob_thresh = prob.sort(descending=True)[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
                prob_thresh.masked_fill_(glance_nums == 0, 100)
                keep_prob = (prob >= prob_thresh.unsqueeze(-1)).to(prob.dtype)

            elif self.glance_strategy == "cmlm":
                prob = torch.randn(oracle.shape, device=tgt_tokens.device, dtype=torch.float)
                prob.masked_fill_(~predict_align_mask, -100)
                glance_nums = (target_length * torch.rand_like(target_length, dtype=torch.float) + 0.5).to(torch.long)
                #prob_thresh = prob.topk(glance_nums.max().clip(min=1))[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
                prob_thresh = prob.sort(descending=True)[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
                prob_thresh.masked_fill_(glance_nums == 0, 100)
                keep_prob = (prob >= prob_thresh.unsqueeze(-1)).to(prob.dtype)

            keep_word_mask = (torch.rand(prev_output_tokens.shape, device=prev_output_tokens.device) < keep_prob).bool()

            glat_prev_output_tokens = prev_output_tokens.masked_fill(keep_word_mask, 0) + oracle.masked_fill(~keep_word_mask, 0)
            glat_tgt_tokens = tgt_tokens

            glat_info = {
                "glat_accu": (same_num.sum() / target_length.sum()).detach(),
                "glat_context_p": glat['context_p'],
                "glat_keep": keep_prob.mean().detach(),
                "matchmask": matchmask,
                "keep_word_mask": keep_word_mask,
                "glat_prev_output_tokens": glat_prev_output_tokens,
            }

            return glat_prev_output_tokens, glat_tgt_tokens, glat_info

        with torch.set_grad_enabled(self.training and sample["update_num"] > self.cfg.dag_freezing_steps):
            outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat, glat_function)

        # DAG loss
        dag_loss, alpha, beta = self._compute_dag_loss_with_alpha_beta(
            outputs["word_ins"].get("out"),
            prev_output_tokens.ne(self.task.tgt_dict.pad()),
            outputs["word_ins"].get("tgt"),
            outputs["word_ins"].get("mask", None),
            outputs["links"],
            name="dag-loss",
            factor=1,
            matchmask=outputs.get('matchmask', None),
            keep_word_mask=outputs.get('keep_word_mask', None),
            model=model
        )

        ####### FastSpeech 2 #######
        if self.cfg.training_strategy == "argmax":
            """
                During training, compute the best alignment a* = argmax P(y, a | x) with viterbi algorithm.
                The input to the TTS model is the hidden states on the best alignment path: z_i = v_{a*_i}.
            """
            with torch.no_grad():
                word_ins_out = outputs["word_ins"]["out"].clone().detach()  # B * L * |V|
                links = outputs["links"].clone().detach()  # B * L * max_trans_len
                _, prelen, _ = links.shape
                nonpad_positions = ~tgt_tokens.eq(model.pad)
                target_length = (nonpad_positions).sum(1)  # [t_1, ..., t_B]
                output_length = prev_output_tokens.ne(model.pad).sum(1)  # [l_1, ..., l_B]

                if self.cfg.torch_dag_logsoftmax_gather:
                    _, match = torch_dag_logsoftmax_gather_inplace(word_ins_out, tgt_tokens.unsqueeze(1).expand(-1, prelen, -1))  # B * L * T
                else:
                    _, match = dag_logsoftmax_gather_inplace(word_ins_out, tgt_tokens.unsqueeze(1).expand(-1, prelen, -1))  # B * L * T
                match = match.transpose(1, 2)  # B * T * L, match[:, i, j] indicates log P(y_i | v_j)
                
                # viterbi for best alignment
                if self.cfg.torch_dag_best_alignment:
                    if model.args.max_transition_length != -1:
                        links = model.restore_valid_links(links)
                    path = torch_dag_best_alignment(match, links, output_length, target_length)
                else:
                    assert model.args.max_transition_length != -1, "cuda dag best alignment does not support max_transition_length=-1. You can use a very large number such as 99999"
                    path = dag_best_alignment(match, links, output_length, target_length) # batch * prelen

            # path: B * L
            # when path[:, i] >= 0, it represents the index of target token aligned with the i-th vertex
            # otherwise, path[:, i] = -1, it represents the i-th vertex is not aligned with any target token
            # below extract the hidden states on the best alignment path
            path[:, 0] = -1  # mask <bos> tokens
            features = outputs["word_ins"]["features"]  # B * L * D
            features_mask = path >= 0
            features_on_path_list = [feature[mask] for feature, mask in zip(features, features_mask)]
            features_on_path = _collate_frames(features_on_path_list)
            input_to_tts = model.adaptor(features_on_path)
            features_padding_mask = lengths_to_padding_mask(features_mask.sum(dim=-1))
        elif self.cfg.training_strategy == "expect":
            """
                During training, compute the posterior probability P(a_i = j | x, y) with forward-backward algorithm, which represents the probability that the i-th target word is aligned with the j-th vertex in DAG.
                The input to the TTS model is the expected hidden states: z_i = \sum_j P(a_i = j | x, y) * v_j.
            """
            features = outputs["word_ins"]["features"]  # B * L * D
            # compute log p(a_i = j | x, y)
            score = (alpha + beta - logsumexp_keepdim(alpha + beta, dim=-1)).exp()
            score.masked_fill_(torch.isnan(score), 0)
            score = score.to(features)  # B * M * L
            expect_features = torch.matmul(score, features)  # B * M * D
            expect_features = expect_features[:, 1:, :]  # remove <bos>
            input_to_tts = model.adaptor(expect_features)
            features_padding_mask = lengths_to_padding_mask(sample["target_text_lengths"] - 1)
    
        # tts forward
        _feat_out, _feat_out_post, _, log_dur_out, pitch_out, energy_out = model.tts(
            input_to_tts, 
            features_padding_mask,
            durations=sample["durations"],
            pitches=sample["pitches"],
            energies=sample["energies"],
        )
        src_mask = lengths_to_mask(sample["target_text_lengths"] - 1)  # -1 to remove <bos> tokens
        tgt_mask = lengths_to_mask(sample["target_audio_lengths"])

        pitches, energies = sample["pitches"], sample["energies"]
        pitch_out, pitches = pitch_out[src_mask], pitches[src_mask]
        energy_out, energies = energy_out[src_mask], energies[src_mask]

        feat_out, feat = _feat_out[tgt_mask], sample["target_audio"][tgt_mask]
        reduction="mean"
        l1_loss = F.l1_loss(feat_out, feat, reduction=reduction)
        if _feat_out_post is not None:
            l1_loss += F.l1_loss(_feat_out_post[tgt_mask], feat, reduction=reduction)

        pitch_loss = F.mse_loss(pitch_out, pitches, reduction=reduction)
        energy_loss = F.mse_loss(energy_out, energies, reduction=reduction)

        log_dur_out = log_dur_out[src_mask]
        dur = sample["durations"].float()
        dur = dur.half() if log_dur_out.type().endswith(".HalfTensor") else dur
        log_dur = torch.log(dur + 1)[src_mask]
        dur_loss = F.mse_loss(log_dur_out, log_dur, reduction=reduction)

        # NOTE: currently not support ctc_loss in FastSpeech 2
        tts_loss = l1_loss + dur_loss + pitch_loss + energy_loss
        ####### FastSpeech 2 #######

        nsentences = dag_loss["nsentences"]
        ntokens = dag_loss["ntokens"]
        nvalidtokens = dag_loss["nvalidtokens"]
        invalid_nsentences = dag_loss["invalid_nsentences"]

        loss = dag_loss["loss"] + tts_loss * self.cfg.tts_loss_weight

        sample_size = 1
        logging_output = {
            "loss": loss.data, 
            "dag-loss": dag_loss["loss"].data,
            "tts-loss": tts_loss.data,
            "l1-loss": l1_loss.data,
            "dur-loss": dur_loss.data,
            "pitch-loss": pitch_loss.data,
            "energy-loss": energy_loss.data,
            "ntokens": ntokens,
            "nvalidtokens": nvalidtokens,
            "nsentences": nsentences,
            "invalid_nsentences": invalid_nsentences,
            "sample_size": sample_size,
            "glat_acc": outputs.get("glat_accu", 0),
            "glat_keep": outputs.get("glat_keep", 0),
        }

        # gpu_tracker.track()
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )  # each batch is 1
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))  # token-level loss

        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nvalidtokens = sum(log.get('nvalidtokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        invalid_nsentences = sum(log.get('invalid_nsentences', 0) for log in logging_outputs)
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))  # token-level loss
        glat_acc = utils.item(sum(log.get("glat_acc", 0) for log in logging_outputs))
        glat_keep = utils.item(sum(log.get("glat_keep", 0) for log in logging_outputs))

        res = {
            "ntokens": utils.item(ntokens),
            "nsentences": utils.item(nsentences),
            "nvalidtokens": utils.item(nvalidtokens),
            "invalid_nsentences": utils.item(invalid_nsentences),
            'tokens_perc': utils.item(nvalidtokens / ntokens),
            'sentences_perc': 1 - utils.item(invalid_nsentences / nsentences),
        }
        res["loss"] = loss / sample_size
        res["glat_acc"] = glat_acc / sample_size
        res["glat_keep"] = glat_keep / sample_size

        for key, value in res.items():
            metrics.log_scalar(
                key, value, sample_size, round=3
            )

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = utils.item(sum(log.get(key, 0) for log in logging_outputs))
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )
