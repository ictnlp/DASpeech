import logging
from pathlib import Path

from fairseq import checkpoint_utils
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.nat.fairseq_nat_model import FairseqNATDecoder
from fairseq.models.speech_to_text.s2t_conformer import (
    S2TConformerModel,
    S2TConformerEncoder,
)

logger = logging.getLogger(__name__)


class S2TConformerNATModel(S2TConformerModel):
    """
    Non-autoregressive Conformer-based Speech-to-text Model.
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.tgt_dict = decoder.dictionary
        self.bos = decoder.dictionary.bos()
        self.eos = decoder.dictionary.eos()
        self.pad = decoder.dictionary.pad()
        self.unk = decoder.dictionary.unk()

    @property
    def allow_length_beam(self):
        return False

    @staticmethod
    def add_args(parser):
        S2TConformerModel.add_args(parser)
        parser.add_argument(
            "--apply-bert-init",
            action="store_true",
            help="use custom param initialization for BERT",
        )
    
    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        decoder = FairseqNATDecoder(args, task.target_dictionary, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward_encoder(self, encoder_inputs):
        return self.encoder(*encoder_inputs)

    def forward_decoder(self, *args, **kwargs):
        return NotImplementedError

    def initialize_output_tokens(self, *args, **kwargs):
        return NotImplementedError

    def forward(self, *args, **kwargs):
        return NotImplementedError