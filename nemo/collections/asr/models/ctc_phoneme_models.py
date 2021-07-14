# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
from typing import Dict, List, Optional

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.per import PER
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.common import tokenizers
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging, model_utils


class EncDecCTCModelPhoneme(EncDecCTCModel):
    """Encoder-decoder CTC-based models that predict phonemes."""

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return []

    def __init__(self, cfg: DictConfig, trainer=None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # Get list of valid phonemes
        if 'phonemes_file' in cfg:
            phonemes_file = cfg.get('phonemes_file')
        else:
            logging.error("ERROR: You must specify a phonemes file in the config.")
            raise ValueError("`cfg` must have `phonemes_file` path to create a tokenizer!")
        phonemes_file = self.register_artifact('phonemes_file', phonemes_file)

        # Create WordTokenizer and override number of classes in the decoder if a placeholder was given
        self.tokenizer = tokenizers.WordTokenizer(vocab_file=phonemes_file)
        vocabulary = self.tokenizer.vocab

        with open_dict(cfg):
            cfg.decoder.vocabulary = ListConfig(list(vocabulary.keys()))

        num_classes = cfg.decoder['num_classes']

        if num_classes < 1:
            logging.info(
                "\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                    num_classes, len(vocabulary)
                )
            )
            cfg.decoder["num_classes"] = len(vocabulary)

        super().__init__(cfg=cfg, trainer=trainer)

        # Set up PER metric (override WER from parent class)
        self._wer = PER(
            tokenizer=self.tokenizer,
            batch_dim_index=0,
            ctc_decode=True,
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", False),
        )

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        # Automatically inject args from model config to dataloader config
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='sample_rate')

        shuffle = config['shuffle']

        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` was None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = audio_to_text_dataset.get_tarred_phoneme_dataset(
                config=config,
                tokenizer=self.tokenizer,
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
                augmentor=augmentor,
            )
            shuffle = False
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None

            dataset = audio_to_text_dataset.get_phoneme_dataset(
                config=config, tokenizer=self.tokenizer, augmentor=augmentor
            )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def change_vocabulary(self, new_phonemes_file: str):
        """
        Changes vocabulary of the tokenizer used during CTC decoding process.
        Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on a data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        Args:
            new_phonemes_file: Path to the new phonemes file.

        Returns: None
        """
        phonemes_file = self.register_artifact('phonemes_file', new_phonemes_file)

        # Create new tokenizer
        del self.tokenizer
        self.tokenizer = tokenizers.WordTokenizer(vocab_file=phonemes_file)

        # Set the new vocabulary
        vocabulary = self.tokenizer.vocab
        decoder_config = copy.deepcopy(self.decoder.to_config_dict())
        decoder_config.vocabulary = ListConfig(list(vocabulary.keys()))

        decoder_num_classes = decoder_config['num_classes']

        # Override number of classes if placeholder provided
        logging.info(
            "\nReplacing old number of classes ({}) with new number of classes - {}".format(
                decoder_num_classes, len(vocabulary)
            )
        )
        decoder_config['num_classes'] = len(vocabulary)

        del self.decoder
        self.decoder = EncDecCTCModelPhoneme.from_config_dict(decoder_config)
        del self.loss
        self.loss = CTCLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
        )
        self._wer = PER(
            tokenizer=self.tokenizer,
            batch_dim_index=0,
            ctc_decode=True,
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", False),
        )

        # Update config
        OmegaConf.set_struct(self._cfg.decoder, False)
        self._cfg.decoder = decoder_config
        OmegaConf.set_struct(self._cfg.decoder, True)

        logging.info(f"Changed tokenizer vocabulary to {self.decoder.vocabulary}.")
