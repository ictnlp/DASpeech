import torch
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from fairseq.data import data_utils as fairseq_data_utils
from fairseq.data import ConcatDataset, Dictionary, FairseqDataset
from fairseq.data.audio.speech_to_text_dataset import (
    _collate_frames,
    _is_int_or_np_int,
    SpeechToTextDatasetCreator,
)
from fairseq.data.audio.audio_utils import get_features_or_waveform
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform
from fairseq.data.audio.waveform_transforms import CompositeAudioWaveformTransform
from DASpeech.datasets.nat_speech_to_speech_data_cfg import NATS2SDataConfig

logger = logging.getLogger(__name__)


@dataclass
class NATSpeechToSpeechDatasetItem(object):
    index: int
    source: torch.Tensor
    target_text: Optional[torch.Tensor] = None
    target_audio: Optional[torch.Tensor] = None
    duration: Optional[torch.Tensor] = None
    pitch: Optional[torch.Tensor] = None
    energy: Optional[torch.Tensor] = None


# NOTE: currently we only support mel-spectrogram as target rather than discrete units
class NATSpeechToSpeechDataset(FairseqDataset):
    def __init__(
        self,
        split: str,
        is_train_split: bool,
        cfg: NATS2SDataConfig,
        src_audio_paths: List[str],
        src_n_frames: List[int],
        tgt_audio_paths: Optional[List[str]] = None,
        tgt_n_frames: Optional[List[int]] = None,
        tgt_texts: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        tgt_dict: Optional[Dictionary] = None,
        pre_tokenizer=None,
        bpe_tokenizer=None,
        n_frames_per_step: int = 1,
        durations: Optional[List[List[int]]] = None,
        pitches: Optional[List[str]] = None,
        energies: Optional[List[str]] = None,
    ):
        self.split, self.is_train_split = split, is_train_split
        self.cfg = cfg
        self.src_audio_paths, self.src_n_frames = src_audio_paths, src_n_frames
        self.tgt_audio_paths, self.tgt_n_frames = tgt_audio_paths, tgt_n_frames
        self.tgt_texts = tgt_texts
        self.ids = ids
        
        self.n_samples = len(src_n_frames)
        self.shuffle = cfg.shuffle if is_train_split else False
        
        # NOTE: currently not support dataset transforms
        self.source_feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            self.cfg.get_source_feature_transforms(split, is_train_split)
        )
        self.source_waveform_transforms = CompositeAudioWaveformTransform.from_config_dict(
            self.cfg.get_source_waveform_transforms(split, is_train_split)
        )
        self.target_feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            self.cfg.get_target_feature_transforms(split, is_train_split)
        )
        self.target_waveform_transforms = CompositeAudioWaveformTransform.from_config_dict(
            self.cfg.get_target_waveform_transforms(split, is_train_split)
        )

        # NOTE: currently not support raw audio input
        assert not self.cfg.use_audio_input
        
        self.tgt_dict = tgt_dict
        self.pre_tokenizer = pre_tokenizer
        self.bpe_tokenizer = bpe_tokenizer
        self.tgt_lens = self.get_tgt_lens_and_check_oov()

        # NOTE: n_frames_per_step is used for target audio rather than source audio
        self.n_frames_per_step = n_frames_per_step  
        
        self.durations = durations
        self.pitches = pitches
        self.energies = energies

        logger.info(self.__repr__())

    def get_tgt_lens_and_check_oov(self):
        if self.tgt_texts is None:
            return [0 for _ in range(self.n_samples)]
        tgt_lens = []
        n_tokens, n_oov_tokens = 0, 0
        for i in range(self.n_samples):
            tokenized = self.get_tokenized_tgt_text(i).split(" ")
            oov_tokens = [
                t
                for t in tokenized
                if self.tgt_dict.index(t) == self.tgt_dict.unk_index
            ]
            n_tokens += len(tokenized)
            n_oov_tokens += len(oov_tokens)
            tgt_lens.append(len(tokenized))
        logger.info(f"'{self.split}' has {n_oov_tokens / n_tokens * 100:.2f}% OOV")
        return tgt_lens

    def __repr__(self):
        return (
            self.__class__.__name__
            + f'(split="{self.split}", n_samples={self.n_samples:_}, '
            f"prepend_tgt_lang_tag={self.cfg.prepend_tgt_lang_tag}, "
            f"n_frames_per_step={self.n_frames_per_step}, "
            f"shuffle={self.shuffle}, "
            f"source_feature_transforms={self.source_feature_transforms}, "
            f"source_waveform_transforms={self.source_waveform_transforms}, "
            f"target_feature_transforms={self.target_feature_transforms}, "
            f"target_waveform_transforms={self.target_waveform_transforms}, "
        )
    
    @classmethod
    def tokenize(cls, tokenizer, text: str):
        return text if tokenizer is None else tokenizer.encode(text)

    def get_tokenized_tgt_text(self, index: Union[int, List[int]]):
        if _is_int_or_np_int(index):
            text = self.tgt_texts[index]
        else:
            text = " ".join([self.tgt_texts[i] for i in index])

        text = self.tokenize(self.pre_tokenizer, text)
        text = self.tokenize(self.bpe_tokenizer, text)
        return text

    def pack_frames(self, feature: torch.Tensor):
        if self.n_frames_per_step == 1:
            return feature
        n_packed_frames = feature.shape[0] // self.n_frames_per_step
        feature = feature[: self.n_frames_per_step * n_packed_frames]
        return feature.reshape(n_packed_frames, -1)

    def _get_source_audio(self, index: int) -> torch.Tensor:
        """
        Gives source audio for given index with any relevant transforms applied.
        """
        source = get_features_or_waveform(
            self.src_audio_paths[index],
            waveform_transforms=self.source_waveform_transforms,
        )
        if self.source_feature_transforms is not None:
            source = self.source_feature_transforms(source)
        source = torch.from_numpy(source).float()
        return source

    def _get_target_audio(self, index: int) -> torch.Tensor:
        """
        Gives target audio for given index with any relevant transforms applied.
        """
        target = get_features_or_waveform(
            self.tgt_audio_paths[index],
            waveform_transforms=self.target_waveform_transforms,
        )
        if self.target_feature_transforms is not None:
            target = self.target_feature_transforms(target)
        target = torch.from_numpy(target).float()
        return target

    def __getitem__(self, index: int) -> NATSpeechToSpeechDatasetItem:
        # source audio
        source = self._get_source_audio(index)

        # target text
        # NOTE: append eos and prepend bos
        target_text = None
        if self.tgt_texts is not None:
            tokenized = self.get_tokenized_tgt_text(index)
            target_text = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True,
            ).long()
            bos = torch.LongTensor([self.tgt_dict.bos()])
            target_text = torch.cat((bos, target_text), 0)

        # target audio
        target_audio = None
        if self.tgt_audio_paths is not None:
            target_audio = self._get_target_audio(index)
            target_audio = self.pack_frames(target_audio)

        # variations
        duration, pitch, energy = None, None, None
        if self.durations is not None:
            duration = torch.tensor(
                self.durations[index] + [0], dtype=torch.long  # pad 0 for EOS
            )
        if self.pitches is not None:
            pitch = get_features_or_waveform(self.pitches[index])
            pitch = torch.from_numpy(
                np.concatenate((pitch, [0]))  # pad 0 for EOS
            ).float()
        if self.energies is not None:
            energy = get_features_or_waveform(self.energies[index])
            energy = torch.from_numpy(
                np.concatenate((energy, [0]))  # pad 0 for EOS
            ).float()

        return NATSpeechToSpeechDatasetItem(
            index=index, source=source, target_text=target_text, target_audio=target_audio,
            duration=duration, pitch=pitch, energy=energy,
        )

    def collater(
        self, samples: List[NATSpeechToSpeechDatasetItem], return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        # source audio
        sources = [x.source for x in samples]
        frames = _collate_frames(sources, self.cfg.use_audio_input)
        # sort samples by descending number of frames
        n_frames = torch.tensor([x.size(0) for x in sources], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        # target text
        target_text, target_text_lengths, ntokens_text = None, None, None
        if self.tgt_texts is not None:
            target_text = fairseq_data_utils.collate_tokens(
                [x.target_text for x in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            ).index_select(0, order)
            target_text_lengths = torch.tensor(
                [x.target_text.size(0) for x in samples], dtype=torch.long
            ).index_select(0, order)
            ntokens_text = sum(x.target_text.size(0) for x in samples)

        # target audio
        target_audio, target_audio_lengths, ntokens_audio = None, None, None
        if self.tgt_audio_paths is not None:
            target_audio = _collate_frames(
                [x.target_audio for x in samples], is_audio_input=False
            ).index_select(0, order)
            target_audio_lengths = torch.tensor(
                [x.target_audio.size(0) for x in samples], dtype=torch.long
            ).index_select(0, order)
            ntokens_audio = sum(x.target_audio.size(0) for x in samples)

        # variations
        durations, pitches, energies = None, None, None
        if self.durations is not None:
            durations = fairseq_data_utils.collate_tokens(
                [x.duration for x in samples], 0
            ).index_select(0, order)
        if self.pitches is not None:
            pitches = _collate_frames([x.pitch for x in samples], True)
            pitches = pitches.index_select(0, order)
        if self.energies is not None:
            energies = _collate_frames([x.energy for x in samples], True)
            energies = energies.index_select(0, order)

        net_input = {
            "src_tokens": frames,
            "src_lengths": n_frames,
        }
        out = {
            "id": indices,
            "net_input": net_input,
            "target_text": target_text,
            "target_text_lengths": target_text_lengths,
            "target_audio": target_audio,
            "target_audio_lengths": target_audio_lengths,
            "durations": durations,
            "pitches": pitches,
            "energies": energies,
            "ntokens_text": ntokens_text,
            "ntokens_audio": ntokens_audio,
            "nsentences": len(samples),
        }
        if return_order:
            out["order"] = order
        return out

    def __len__(self):
        return self.n_samples

    def num_tokens(self, index):
        return self.src_n_frames[index]
    
    def size(self, index):
        return self.src_n_frames[index], self.tgt_lens[index], self.tgt_n_frames[index]
    
    @property
    def sizes(self):
        return np.array(self.src_n_frames)
    
    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        order.append([-n for n in self.src_n_frames])
        return np.lexsort(order)

    def prefetch(self, indices):
        raise False



class NATSpeechToSpeechDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_SRC_AUDIO, KEY_SRC_N_FRAMES = "id", "src_audio", "src_n_frames"
    # optional columns
    KEY_TGT_AUDIO, KEY_TGT_N_FRAMES = "tgt_audio", "tgt_n_frames"
    KEY_TGT_TEXT = "tgt_text"
    KEY_DURATION, KEY_PITCH, KEY_ENERGY = "duration", "pitch", "energy"

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        cfg: NATS2SDataConfig,
        tgt_dict: Dictionary = None,
        n_frames_per_step: int = 1,
    ) -> NATSpeechToSpeechDataset:
        ids = [s[cls.KEY_ID] for s in samples]
        src_audio_paths = [s[cls.KEY_SRC_AUDIO] for s in samples]
        src_n_frames = [int(s[cls.KEY_SRC_N_FRAMES]) for s in samples]
        tgt_audio_paths = [s.get(cls.KEY_TGT_AUDIO, None) for s in samples]
        tgt_n_frames = [int(s.get(cls.KEY_TGT_N_FRAMES, 0)) for s in samples]
        tgt_texts = [s.get(cls.KEY_TGT_TEXT, "") for s in samples]

        tgt_audio_paths = None if any(tgt is None for tgt in tgt_audio_paths) else tgt_audio_paths
        durations = [s.get(cls.KEY_DURATION, None) for s in samples]
        durations = [
            None if dd is None else [int(d) for d in dd.split(" ")] for dd in durations
        ]
        durations = None if any(dd is None for dd in durations) else durations

        pitches = [s.get(cls.KEY_PITCH, None) for s in samples]
        pitches = None if any(pp is None for pp in pitches) else pitches

        energies = [s.get(cls.KEY_ENERGY, None) for s in samples]
        energies = None if any(ee is None for ee in energies) else energies

        ds = NATSpeechToSpeechDataset(
            split=split_name,
            is_train_split=is_train_split,
            cfg=cfg,
            src_audio_paths=src_audio_paths,
            src_n_frames=src_n_frames,
            tgt_audio_paths=tgt_audio_paths,
            tgt_n_frames=tgt_n_frames,
            tgt_texts=tgt_texts,
            ids=ids,
            tgt_dict=tgt_dict,
            n_frames_per_step=n_frames_per_step,
            durations=durations,
            pitches=pitches,
            energies=energies,
        )

        return ds

    @classmethod
    def from_tsv(
        cls,
        root: str,
        cfg: NATS2SDataConfig,
        splits: str,
        is_train_split: bool,
        tgt_dict: Dictionary = None,
        n_frames_per_step: int = 1,
    ) -> NATSpeechToSpeechDataset:
        datasets = []
        for split in splits.split(","):
            samples = SpeechToTextDatasetCreator._load_samples_from_tsv(root, split)
            ds = cls._from_list(
                split_name=split,
                is_train_split=is_train_split,
                samples=samples,
                cfg=cfg,
                tgt_dict=tgt_dict,
                n_frames_per_step=n_frames_per_step,
            )
            datasets.append(ds)
        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
