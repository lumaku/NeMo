# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.nlp.data import TextNormalizationDataset, TextNormalizationTestDataset
from nemo.collections.nlp.data.text_normalization.text_normalization_dataset import tag_labels
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.models.text_normalization.modules import (
    Attention,
    DecoderAttentionRNN,
    EncoderDecoder,
    EncoderRNN,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import NeuralType
from nemo.utils import logging

__all__ = ['TextNormalizationModel']


class TextNormalizationModel(NLPModel):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return None

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self._tokenizer_context = self._setup_tokenizer(cfg.tokenizer_context)
        self._tokenizer_encoder = self._setup_tokenizer(cfg.tokenizer_encoder)
        self._tokenizer_decoder = self._setup_tokenizer(cfg.tokenizer_decoder)
        super().__init__(cfg=cfg, trainer=trainer)

        self.context_embedding = nn.Embedding(
            self._tokenizer_context.vocab_size, cfg.context.embedding_size, padding_idx=self._tokenizer_context.pad_id
        )
        self.context_embedding.weight.data.normal_(0, 0.1)
        self.context_encoder = EncoderRNN(
            input_size=cfg.context.embedding_size,
            hidden_size=cfg.context.hidden_size,
            num_layers=cfg.context.num_layers,
            dropout=cfg.context.dropout,
        )
        self.tagger_decoder = nn.GRU(
            input_size=cfg.tagger.embedding_size,
            hidden_size=cfg.tagger.hidden_size,
            num_layers=cfg.tagger.num_layers,
            dropout=cfg.tagger.dropout,
            batch_first=True,
        )
        self.context_project = nn.Linear(cfg.context.hidden_size, cfg.tagger.embedding_size)
        self.tagger_output_project = nn.Linear(cfg.tagger.hidden_size, cfg.tagger.num_classes)

        self.seq2seq = EncoderDecoder(
            EncoderRNN(
                input_size=cfg.seq_encoder.embedding_size,
                hidden_size=cfg.seq_encoder.hidden_size,
                num_layers=cfg.seq_encoder.num_layers,
                dropout=cfg.seq_encoder.dropout,
            ),
            DecoderAttentionRNN(
                attention=Attention(
                    attention_hidden_size=cfg.seq_decoder.attention_size,
                    encoder_hidden_size=cfg.seq_encoder.hidden_size,
                    decoder_hidden_size=cfg.seq_decoder.hidden_size,
                ),
                embed_size=cfg.seq_decoder.embedding_size,
                hidden_size=cfg.seq_decoder.hidden_size,
                num_layers=cfg.seq_decoder.num_layers,
                dropout=cfg.seq_decoder.dropout,
                num_classes=self._tokenizer_decoder.vocab_size,
            ),
            nn.Embedding(
                self._tokenizer_encoder.vocab_size,
                cfg.seq_encoder.embedding_size,
                padding_idx=self._tokenizer_encoder.pad_id,
            ),
            nn.Embedding(
                self._tokenizer_decoder.vocab_size,
                cfg.seq_decoder.embedding_size,
                padding_idx=self._tokenizer_decoder.pad_id,
            ),
            teacher_forcing=cfg.seq_decoder.teacher_forcing,
        )

        self.seq2seq_loss = CrossEntropyLoss(logits_ndim=3)
        self.tagger_loss = CrossEntropyLoss(logits_ndim=3)

        self.classification_report = ClassificationReport(
            num_classes=cfg.tagger.num_classes, mode='micro', dist_sync_on_step=True
        )

    def tagger_forward(self, context_output):
        context_to_tagger = self.context_project(context_output[:, :, self._cfg.context.hidden_size :])
        context_to_tagger = nn.functional.tanh(context_to_tagger)
        tagger_output, _ = self.tagger_decoder(context_to_tagger)
        tagger_logits = self.tagger_output_project(tagger_output)
        return tagger_logits

    def get_context(self, context_ids, len_context):
        context_embedding = self.context_embedding(context_ids)
        context_output, _ = self.context_encoder(input_seqs=context_embedding, input_lengths=len_context)
        left_context = context_output[:, :, : self._cfg.context.hidden_size].view(-1, self._cfg.context.hidden_size)
        right_context = context_output[:, :, self._cfg.context.hidden_size :].view(-1, self._cfg.context.hidden_size)
        return context_output, left_context, right_context

    # @typecheck()
    def forward(
        self,
        context_ids,
        len_context,
        input_ids,
        len_input,
        output_ids,
        l_context_ids,
        r_context_ids,
        max_len: Optional[int] = None,
    ):
        context_output, left_context, right_context = self.get_context(context_ids, len_context)

        batch_size, max_context_length = context_ids.shape
        context_lr = torch.cat(
            [
                left_context[torch.arange(batch_size).to(self._device) * max_context_length + l_context_ids],
                right_context[torch.arange(batch_size).to(self._device) * max_context_length + r_context_ids],
            ],
            dim=-1,
        ).unsqueeze(0)
        tagger_logits = self.tagger_forward(context_output)
        seq_logits = self.seq2seq(
            src=input_ids, trg=output_ids, src_lengths=len_input, decoder_init_hidden=context_lr, max_len=max_len
        )

        return tagger_logits, seq_logits

    def _setup_tokenizer(self, cfg: DictConfig):
        """Instantiates tokenizer based on config and registers tokenizer artifacts.

           If model is being restored from .nemo file then the tokenizer.vocab_file will
           be used (if it exists).

           Otherwise, we will use the vocab file provided in the config (if it exists).

           Finally, if no vocab file is given (this happens frequently when using HF),
           we will attempt to extract the vocab from the tokenizer object and then register it.

        Args:
            cfg (DictConfig): Tokenizer config
        """
        vocab_file = None
        if cfg.vocab_file:
            vocab_file = self.register_artifact(config_path='tokenizer.vocab_file', src=cfg.vocab_file)
        tokenizer = get_tokenizer(
            tokenizer_name=cfg.tokenizer_name,
            vocab_file=vocab_file,
            special_tokens=OmegaConf.to_container(cfg.special_tokens) if cfg.special_tokens else None,
            tokenizer_model=self.register_artifact(config_path='tokenizer.tokenizer_model', src=cfg.tokenizer_model),
        )

        # if vocab_file is None:
        #     # when there is no vocab file we try to get the vocab from the tokenizer and register it
        #     self._register_vocab_from_tokenizer(vocab_file_config_path='tokenizer.vocab_file', cfg=cfg)
        return tokenizer

    def training_step(self, batch, batch_idx):
        (
            context_ids,
            tag_ids,
            len_context,
            input_ids,
            len_input,
            output_ids,
            len_output,
            l_context_ids,
            r_context_ids,
            example_ids,
        ) = batch
        bs, max_context_length = context_ids.shape
        _, max_target_length = output_ids.shape
        tagger_logits, seq_logits = self.forward(
            context_ids=context_ids,
            len_context=len_context,
            input_ids=input_ids,
            len_input=len_input,
            output_ids=output_ids,
            l_context_ids=l_context_ids,
            r_context_ids=r_context_ids,
        )
        tagger_loss_mask = torch.arange(max_context_length).to(self._device).expand(
            bs, max_context_length
        ) < len_context.unsqueeze(1)
        tagger_loss = self.tagger_loss(logits=tagger_logits, labels=tag_ids, loss_mask=tagger_loss_mask)
        seq_loss_mask = torch.arange(max_target_length).to(self._device).expand(
            bs, max_target_length
        ) < len_output.unsqueeze(1)
        seq_loss = self.seq2seq_loss(logits=seq_logits, labels=output_ids, loss_mask=seq_loss_mask)
        loss = tagger_loss + seq_loss
        lr = self._optimizer.param_groups[0]['lr']
        self.log('train_loss', loss)
        self.log('lr', lr, prog_bar=True)
        return {'loss': loss, 'lr': lr}

    def validation_step(self, batch, batch_idx):
        if self.trainer.testing:
            prefix = 'test'
        else:
            prefix = 'val'

        if prefix == 'val':
            (
                context_ids,
                tag_ids,
                len_context,
                input_ids,
                len_input,
                output_ids,
                len_output,
                l_context_ids,
                r_context_ids,
                example_ids,
            ) = batch
        else:
           
            (
                context_ids,
                len_context,
                example_ids,
            ) = batch 

        bs, max_context_length = context_ids.shape

        if prefix == "val":
            _, max_target_length = output_ids.shape
            tagger_logits, seq_logits = self.forward(
                context_ids=context_ids,
                len_context=len_context,
                input_ids=input_ids,
                len_input=len_input,
                output_ids=output_ids,
                l_context_ids=l_context_ids,
                r_context_ids=r_context_ids,
            )
            tagger_loss_mask = torch.arange(max_context_length).to(self._device).expand(
                bs, max_context_length
            ) < len_context.unsqueeze(1)
            tagger_loss = self.tagger_loss(logits=tagger_logits, labels=tag_ids, loss_mask=tagger_loss_mask)
            tagger_logits = tagger_logits[tagger_loss_mask]
            tag_ids = torch.masked_select(tag_ids, tagger_loss_mask)
            tag_preds = torch.argmax(tagger_logits, axis=-1)

            seq_loss_mask = torch.arange(max_target_length).to(self._device).expand(
                bs, max_target_length
            ) < len_output.unsqueeze(1)
            seq_loss = self.seq2seq_loss(logits=seq_logits, labels=output_ids, loss_mask=seq_loss_mask)
            loss = tagger_loss + seq_loss



        # compute sentence accuracy

        input_ids, tag_preds, seq_preds, row_index, left_column_index, right_column_index = self.inference(
            example_ids, self._cfg.seq_decoder.max_len, self._device, context_ids, len_context, bs, max_context_length
        )

        preds = self.assemble_output(
            context_ids=context_ids,
            seq_preds=seq_preds,
            tag_preds=tag_preds,
            example_ids=example_ids,
            row_index=row_index,
            left_column_index=left_column_index,
            right_column_index=right_column_index,
        )

        if prefix == 'test':
            return {f'{prefix}_preds': preds}
        else:
            return {f'{prefix}_loss': loss, f'{prefix}_preds': preds}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        outputs: list of individual outputs of each validation step.
        """
        
        if self.trainer.testing:
            prefix = 'test'
        else:
            prefix = 'val'

        if prefix == 'val':
            avg_loss = torch.stack([x[f'{prefix}_loss'] for x in outputs]).mean()
            self.log(f'{prefix}_loss', avg_loss)

        all_preds = {}
        for x in outputs:
            all_preds.update(x[f'{prefix}_preds'])

        
        eval_dataset = self._test_dl.dataset if self.trainer.testing else self._validation_dl.dataset
        accuracy = eval_dataset.evaluate(all_preds)

        logging.info(f'{prefix}_accuracy {accuracy}')
        self.log(f'{prefix}_accuracy', accuracy)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        """
        Called at test time to aggregate outputs.
        outputs: list of individual outputs of each validation step.
        """
        return self.validation_epoch_end(outputs)

    def assemble_output(
        self, context_ids, seq_preds, tag_preds, example_ids, row_index, left_column_index, right_column_index
    ):
        preds = defaultdict(list)

        context_ids = tensor2list(context_ids)
        tag_preds = tensor2list(tag_preds)
        if seq_preds is not None:
            seq_preds = tensor2list(seq_preds)
            row_index = tensor2list(row_index)
            left_column_index = tensor2list(left_column_index)
            right_column_index = tensor2list(right_column_index)
        example_ids = tensor2list(example_ids)

        decoder_counter = 0

        def decode(prev_tag, i, start_index, j, decoder_counter):
            if prev_tag == "O":
                preds[example_id].append(self._tokenizer_context.ids_to_text(context_ids[i][start_index:j]))
            elif prev_tag == "B":
                preds[example_id].append(self._tokenizer_decoder.ids_to_text(seq_preds[decoder_counter]))
                decoder_counter += 1
            return decoder_counter

        for i in range(len(context_ids)):
            example_id = example_ids[i]
            if example_id in preds:
                continue

            prev_tag = None
            start_index = 0

            for j, tag in enumerate(tag_preds[i]):

                if context_ids[i][j] == self._tokenizer_context.bos_id:
                    continue
                if context_ids[i][j] == self._tokenizer_context.eos_id:
                    break

                if tag == tag_labels['O-I']:
                    decoder_counter = decode(prev_tag, i, start_index, j, decoder_counter)
                    prev_tag = 'O'
                    start_index = j
                elif tag == tag_labels['O-M']:
                    if prev_tag == 'O':
                        continue
                    else:
                        decoder_counter = decode(prev_tag, i, start_index, j, decoder_counter)
                        prev_tag = 'O'
                        start_index = j
                elif tag == tag_labels['B-I']:

                    decoder_counter = decode(prev_tag, i, start_index, j, decoder_counter)
                    prev_tag = 'B'
                    start_index = j
                else:
                    if prev_tag == 'B':
                        continue
                    else:

                        decoder_counter = decode(prev_tag, i, start_index, j, decoder_counter)
                        prev_tag = 'B'
                        start_index = j

            decoder_counter = decode(prev_tag, i, start_index, j, decoder_counter)

        return preds

    @torch.no_grad()
    def infer(self, cfg: DictConfig, max_output_length=10):
        # store predictions for all queries in a single list
        all_preds = defaultdict(list)
        mode = self.training
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Switch model to evaluation modef
            self.eval()
            self.to(device)
            infer_datalayer = self._setup_dataloader_from_config(cfg=cfg, mode="infer")

            for batch in tqdm(infer_datalayer):
                (
                    context_ids,
                    len_context,
                    example_ids,
                ) = batch
                # only use context_ids, len_context, example_ids
                context_ids = context_ids.to(device)
                len_context = len_context.to(device)
                bs, max_context_length = context_ids.shape

                input_ids, tag_preds, seq_preds, row_index, left_column_index, right_column_index = self.inference(
                    example_ids, max_output_length, device, context_ids, len_context, bs, max_context_length
                )

                # assemble all outputs
                # use context_ids, context_ids, seq_preds, tag_preds, example_ids, row_index, left_column_index, right_column_index
                preds = self.assemble_output(
                    context_ids=context_ids,
                    seq_preds=seq_preds,
                    tag_preds=tag_preds,
                    example_ids=example_ids,
                    row_index=row_index,
                    left_column_index=left_column_index,
                    right_column_index=right_column_index,
                )
                all_preds.update(preds)

        finally:
            # set mode back to its original value
            self.train(mode=mode)
        return all_preds

    def inference(self, example_ids, max_output_length, device, context_ids, len_context, bs, max_context_length):
        # get context encoding, and left and right context embeddings
        context_output, left_context, right_context = self.get_context(context_ids, len_context)

        # get tagger logits
        tagger_logits = self.tagger_forward(context_output)

        tagger_loss_mask = torch.arange(max_context_length).to(device).expand(bs, max_context_length) < len_context.to(
            device
        ).unsqueeze(1)
        tagger_loss_mask[:, 0] = False  # do not allow first token since this should correspond to bos
        tagger_loss_mask[:, -1] = False  # do not allow last token since this should correspond to eos
        tag_preds = torch.argmax(tagger_logits, axis=-1)

        decoder_counter = 0
        row_index = []
        right_column_index = []
        left_column_index = []

        def register_lr(prev_tag, i, start_index, j):
            if prev_tag:
                row_index.append(i)
                left_column_index.append(start_index - 1)
                right_column_index.append(j)

        for i in range(len(tag_preds)):
            prev_tag = None
            start_index = 0

            for j, tag in enumerate(tag_preds[i]):

                if j == 0:
                    continue
                if j >= len_context[i] - 1:
                    register_lr(prev_tag, i, start_index, j)
                    prev_tag = None
                    break

                if tag == tag_labels['B-I']:
                    register_lr(prev_tag, i, start_index, j)
                    prev_tag = 'B'
                    start_index = j
                elif tag == tag_labels['B-M']:
                    if prev_tag == 'B':
                        continue
                    else:
                        register_lr(prev_tag, i, start_index, j)
                        prev_tag = 'B'
                        start_index = j
                else:
                    register_lr(prev_tag, i, start_index, j)
                    prev_tag = None

        row_index = torch.LongTensor(row_index).to(device)
        right_column_index = torch.LongTensor(right_column_index).to(device)
        left_column_index = torch.LongTensor(left_column_index).to(device)

        ############

        # semiotic_tag_preds = torch.logical_or(tag_preds == tag_labels['B-I'], tag_preds == tag_labels['B-M'])
        # semiotic_tag_preds.masked_fill_(
        #             tagger_loss_mask == False, False
        #         )  # do not consider preds outside of maskA

        # l_context_ids = torch.where(
        #             torch.logical_and(semiotic_tag_preds[:, :-1] == False, semiotic_tag_preds[:, 1:] == True)
        #         )
        # r_context_ids = torch.where(
        #             torch.logical_and(semiotic_tag_preds[:, 1:] == False, semiotic_tag_preds[:, :-1] == True)
        #         )
        # row_index = l_context_ids[0]
        # left_column_index = l_context_ids[1]
        # right_column_index = r_context_ids[1] + 1
        # r_context_ids = (row_index, right_column_index + 1)

        ######

        assert (
            torch.all(left_column_index < right_column_index),
            "left context start needs to be smaller than right context end",
        )

        context_lr = torch.cat(
            [
                left_context[row_index * max_context_length + left_column_index],
                right_context[row_index * max_context_length + right_column_index],
            ],
            dim=-1,
        ).unsqueeze(0)

        input_ids = []
        for i in range(len(row_index)):
            batch_id = row_index[i]
            input_str = self._tokenizer_context.ids_to_text(
                tensor2list(context_ids[batch_id][left_column_index[i] + 1 : right_column_index[i]])
            )
            input_ids.append([self._tokenizer_encoder.bos_id] + self._tokenizer_encoder.text_to_ids(input_str))

        len_input = [len(x) for x in input_ids]

        if not input_ids:
            return None, tag_preds, None, row_index, left_column_index, right_column_index

        max_input_length = max(len_input)
        len_input = torch.LongTensor(len_input).to(device)
        for i in range(len(input_ids)):
            pad_width = max_input_length - len(input_ids[i])
            input_ids[i] = np.pad(
                input_ids[i], pad_width=[0, pad_width], constant_values=self._tokenizer_encoder.pad_id
            )

        input_ids = torch.LongTensor(input_ids).to(device)
        decoder_start = (
            torch.tensor([self._tokenizer_decoder.bos_id]).repeat(input_ids.shape[0]).unsqueeze(1).to(device)
        )

        seq_logits = self.seq2seq(
            src=input_ids,
            trg=decoder_start,
            src_lengths=len_input,
            decoder_init_hidden=context_lr,
            max_len=max_output_length,
        )

        seq_preds = torch.argmax(seq_logits, axis=-1)

        return input_ids, tag_preds, seq_preds, row_index, left_column_index, right_column_index

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config or not train_data_config.file:
            logging.info(
                f"Dataloader config or file_path for the train is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, mode="train")

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config or not val_data_config.file:
            logging.info(
                f"Dataloader config or file_path for the validation is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, mode="val")

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if not test_data_config or test_data_config.file is None:
            logging.info(
                f"Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, mode="test")

    def _setup_dataloader_from_config(self, cfg: DictConfig, mode: str):
        input_file = cfg.file
        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f'{input_file} not found! The data should be be stored in TAB-separated files \n\
                "validation_ds.file" and "train_ds.file" for train and evaluation respectively. \n\
                Each line of the files contains text sequences, where words are separated with spaces. \n\
                The label of the example is separated with TAB at the end of each line. \n\
                Each line of the files should follow the format: \n\
                [WORD][SPACE][WORD][SPACE][WORD][...][TAB][LABEL]'
            )

        if mode in ["train", "val"]:
            dataset = TextNormalizationDataset(
                input_file=input_file,
                tokenizer_context=self._tokenizer_context,
                tokenizer_encoder=self._tokenizer_encoder,
                tokenizer_decoder=self._tokenizer_decoder,
                num_samples=cfg.get("num_samples", -1),
                use_cache=self._cfg.dataset.use_cache,
                max_sentence_length=self._cfg.dataset.max_sentence_length
            )
        else:
            dataset = TextNormalizationTestDataset(
                input_file=input_file,
                tokenizer_context=self._tokenizer_context,
                tokenizer_encoder=self._tokenizer_encoder,
                tokenizer_decoder=self._tokenizer_decoder,
                num_samples=cfg.get("num_samples", -1),
                use_cache=self._cfg.dataset.use_cache,
                max_sentence_length=None)
        dl = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
            collate_fn=dataset.collate_fn,
        )
        return dl

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        return result