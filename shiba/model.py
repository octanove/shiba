import json
import math
import os
import urllib
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch

from shiba.codepoint_tokenizer import CodepointTokenizer
from shiba.local_transformer_encoder_layer import LocalTransformerEncoderLayer
from shiba.multi_hashing_embedder import MultiHashingEmbedder

from shiba.position_embedder import PositionEmbedder


class ShibaConfig(SimpleNamespace):
    def to_dict(self):
        out = self.__dict__.copy()
        del out['self']
        return out

    def to_json_string(self):
        return json.dumps(self.to_dict())


class Shiba(torch.nn.Module):

    # defaults modeled after CANINE
    # https://github.com/google-research/language/blob/186ce9002180d0c45bfa2a680085b890c76647dc/language/canine/modeling.py#L40
    def __init__(self, downsampling_rate: int = 4,
                 upsampling_kernel_size: int = 4,
                 embedder_slice_count: int = 8,
                 embedder_bucket_count: int = 16000,
                 hidden_size: int = 768,
                 local_attention_window: int = 128,
                 deep_transformer_stack: Optional[torch.nn.Module] = None,
                 deep_transformer_requires_transpose: bool = True,
                 attention_heads: int = 12,
                 transformer_ff_size: int = 3072,
                 dropout: float = 0.1,
                 activation: str = 'gelu',
                 padding_id: int = 0,
                 max_length: int = 2048,
                 shiba_specific_code: bool = False,
                 deep_transformer_stack_layers: Optional[int] = None):
        super(Shiba, self).__init__()

        self.config = ShibaConfig(**{key: val for key, val in locals().items()
                                         if key in self.__init__.__code__.co_varnames})

        if max_length % downsampling_rate != 0:
            # if this isn't true, padding so we don't miss any characters can bring us over max length
            raise RuntimeError(f"max length must be divisible by downsampling rate, but got "
                               f"{max_length} and {downsampling_rate} respectively")


        activations = {
            'relu': torch.nn.ReLU,
            'gelu': torch.nn.GELU
        }

        if activation not in activations:
            raise RuntimeError(f'activation must be in {set(activations.keys())}, but was {activation}')
        else:
            self.activation = activations[activation]()

        self.dropout = torch.nn.Dropout(p=dropout)

        # "Hash Embedding"
        self.embedder = MultiHashingEmbedder(hidden_size, slice_count=embedder_slice_count,
                                             bucket_count=embedder_bucket_count)

        self.position_embedder = PositionEmbedder(max_length, hidden_size)
        self.embedder_ln = torch.nn.LayerNorm(hidden_size)

        # "Single Local Transformer"
        # note the CANINE paper says "local transformer", but it means "local transformer encoder" just like BERT
        self.local_transformer = LocalTransformerEncoderLayer(hidden_size, attention_heads, dropout=dropout,
                                                              activation=activation,
                                                              dim_feedforward=transformer_ff_size)

        # "Downsample (Strided Convolution) "
        self.downsample_conv = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=downsampling_rate,
                                               stride=downsampling_rate)

        self.downsample_attention_pool = torch.nn.MaxPool1d(kernel_size=downsampling_rate, stride=downsampling_rate)
        self.downsample_ln = torch.nn.LayerNorm(hidden_size)
        self.cls_linear = torch.nn.Linear(hidden_size, hidden_size)

        # "Deep Transformer Stack"
        if deep_transformer_stack is not None:
            if deep_transformer_stack_layers is not None:
                raise RuntimeError('deep_transformer_stack_layers and deep_transformer_stack both provided - please '
                                   'provide only one.')
            # TODO: perform some kind of basic verification that this is actually a torch module that can be used
            # in place of the default deep transformer stack
            self.deep_transformer = deep_transformer_stack
        else:
            layers = deep_transformer_stack_layers if deep_transformer_stack_layers is not None else 12
            self.deep_transformer = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(hidden_size,
                                                                                                 attention_heads,
                                                                                                 dim_feedforward=transformer_ff_size,
                                                                                                 dropout=dropout,
                                                                                                 activation=activation),
                                                                num_layers=layers)
            self.config.deep_transformer_requires_transpose = True

        # "Conv + Single Transformer"
        self.upsample_conv = torch.nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=upsampling_kernel_size, stride=1)
        self.upsample_ln = torch.nn.LayerNorm(hidden_size)
        self.final_transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(hidden_size, attention_heads,
                                             dim_feedforward=transformer_ff_size,
                                             dropout=dropout,
                                             activation=activation),
            1)

        # CLS Token
        self.cls_linear_final = torch.nn.Linear(hidden_size, hidden_size)
        self.cls_activation = torch.nn.Tanh()

    # "Upsampling"
    def _repeat_molecules(self, molecules: torch.Tensor, char_seq_length: int):

        repeated_molecules = molecules.repeat_interleave(self.config.downsampling_rate, axis=1)
        remainder_length = char_seq_length % self.config.downsampling_rate

        # as the canine implementation does, we repeat the last molecule extra times to get to a multiple of 4
        last_molecule = molecules[:, -1:, :]
        last_molecule_repeated = last_molecule.repeat_interleave(remainder_length, axis=1)

        return torch.cat((repeated_molecules, last_molecule_repeated), dim=1)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                predict_indices: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        if input_ids.shape[1] > self.config.max_length:
            raise RuntimeError(f'Input tensor of shape {input_ids.shape} exceeded configured max length'
                               f'{self.config.max_length}')

        if any(input_ids[:, 0:1] != CodepointTokenizer.CLS):
            raise RuntimeError('All input sequences must start wit [CLS] codepoint')


        # https://github.com/google-research/language/blob/186ce9002180d0c45bfa2a680085b890c76647dc/language/canine/modeling.py#L221
        # https://github.com/google-research/language/blob/13dc35ccad77309354ff8ed2950c560c16b083b1/language/canine/bert_modeling.py#L448
        char_embeddings = self.position_embedder(self.embedder(input_ids))

        # https://github.com/google-research/language/blob/186ce9002180d0c45bfa2a680085b890c76647dc/language/canine/modeling.py#L253
        contextualized_chars = self.local_transformer(char_embeddings, attention_mask)  # h_init

        # https://github.com/google-research/language/blob/186ce9002180d0c45bfa2a680085b890c76647dc/language/canine/modeling.py#L287
        cls_embedding = self.dropout(self.cls_linear(contextualized_chars[:, 0:1, :]))
        if self.config.shiba_specific_code:
            # remove the CLS token from the tokens that get downsampled so its information isn't used twice
            contextualized_chars = contextualized_chars[:, 1:, :]
            attention_mask = attention_mask[:, 1:]

            # pad so the convolution can't drop information from final characters
            contextualized_chars = self._pad_to_avoid_missed_characters(contextualized_chars)
            attention_mask = self._pad_to_avoid_missed_characters(attention_mask.unsqueeze(2)).squeeze()

            # note that even with shiba specific code turned off, we don't truncate the last char like CANINE does

        sampleable_characters = contextualized_chars.transpose(1, 2).contiguous()
        sampleable_mask = attention_mask.float()

        molecules = self.downsample_conv(sampleable_characters).transpose(1, 2)  # h_down
        molecules = self.downsample_ln(self.activation(molecules))
        molecules = torch.cat((cls_embedding, molecules), dim=1)

        # unlike CANINE we don't assume a fixed size and truncate, so we have to add to the attention mask for the
        # CLS slot. squeezing and unsqueezing is a fix for https://github.com/pytorch/pytorch/issues/51954
        downsampled_attention_mask = self.downsample_attention_pool(sampleable_mask.unsqueeze(0)).squeeze(0)
        molecule_attention_mask = torch.nn.functional.pad(downsampled_attention_mask.bool(), (1, 0), value=True)

        # https://github.com/google-research/language/blob/186ce9002180d0c45bfa2a680085b890c76647dc/language/canine/modeling.py#L343
        if self.config.deep_transformer_requires_transpose:
            # TODO: if we switch out the deep transformer to something that calls its attention mask
            # anything other than "src_key_padding_mask" this will break
            contextualized_molecules = self.deep_transformer(molecules.transpose(0, 1),
                                                        src_key_padding_mask=molecule_attention_mask).transpose(0, 1)  # h`_down
        else:
            contextualized_molecules = self.deep_transformer(molecules, src_key_padding_mask=molecule_attention_mask)

        # https://github.com/google-research/language/blob/186ce9002180d0c45bfa2a680085b890c76647dc/language/canine/modeling.py#L371
        molecules_without_cls = contextualized_molecules[:, 1:, :]  # remove CLS to avoid upsampling it
        repeated_molecules = self._repeat_molecules(molecules_without_cls, contextualized_chars.shape[1])

        # https://github.com/google-research/language/blob/186ce9002180d0c45bfa2a680085b890c76647dc/language/canine/modeling.py#L468
        concatenated = torch.cat((contextualized_chars, repeated_molecules), dim=2)
        concatenated = self._pad_for_convolution_to_same_length(concatenated, self.upsample_conv)
        upsampled_embeddings = self.activation(self.upsample_conv(concatenated.transpose(1, 2).contiguous()).
                                               transpose(1, 2))
        upsampled_embeddings = self.dropout(self.upsample_ln(upsampled_embeddings))  # h_up

        if predict_indices is not None:
            # this is MLM of some kind - we don't need to do the final CLS computation and we can only do the
            # final transformer for the positions we're predicting
            embeddings_for_pred = torch.stack([upsampled_embeddings[i, predict_indices[i], :]
                                               for i in range(upsampled_embeddings.shape[0])])

            # no attention mask because we are presumably not trying to predict padding
            final_embeddings = self.final_transformer(embeddings_for_pred.transpose(0, 1)).transpose(0, 1)
        else:
            # https://github.com/google-research/language/blob/master/language/canine/modeling.py#L551
            contextualized_cls = contextualized_molecules[:, 0:1, :]
            final_cls = self.cls_activation(self.cls_linear_final(contextualized_cls))

            # confusingly, key_padding_mask does for the pytorch transformer what attention_mask does for the
            # local attention implementation (and huggingface/allennlp)
            # also, we drop the first embedding (CLS token) because we're going to use final_cls anyway
            final_embeddings = self.final_transformer(upsampled_embeddings[:, 1:, :].transpose(0, 1),
                                                      src_key_padding_mask=attention_mask[:, 1:]).transpose(0, 1)
            final_embeddings = torch.cat((final_cls, final_embeddings), dim=1)  # replace CLS embedding

        return {
            'embeddings': final_embeddings
        }

    def _pad_to_avoid_missed_characters(self, char_embeddings: torch.Tensor) -> torch.Tensor:
        if char_embeddings.shape[1] % self.config.downsampling_rate == 0:
            return char_embeddings
        else:
            target_length = math.ceil(char_embeddings.shape[1] / self.config.downsampling_rate)\
                            * self.config.downsampling_rate
            total_padding = target_length - char_embeddings.shape[1]
            lhs_padding = math.floor(total_padding / 2)
            rhs_padding = math.ceil(total_padding / 2)
            return torch.nn.functional.pad(char_embeddings, (0, 0, lhs_padding, rhs_padding))

    def _pad_for_convolution_to_same_length(self, hidden_state: torch.Tensor,
                                            convolution: torch.nn.Conv1d) -> torch.Tensor:
        # we have to manually pad, see: https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/2
        # so we solve for total padding from the formula for output length
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        # hidden state has shape [batch_size, sequence_length, embedding_size]
        l = hidden_state.shape[1]
        s = convolution.stride[0]
        d = convolution.dilation[0]
        k = convolution.kernel_size[0]

        total_padding = l * s - l + d * k - d + 1 - s
        lhs_padding = math.floor(total_padding / 2)
        rhs_padding = math.ceil(total_padding / 2)

        return torch.nn.functional.pad(hidden_state, (0, 0, lhs_padding, rhs_padding))


def get_pretrained_state_dict():
    download_url = 'https://storage.googleapis.com/shiba.octanove.com/published_checkpoints/shiba_check45k.pt'
    save_location = Path.home() / '.shiba' / 'pretrained_shiba.pt'

    if not save_location.parent.exists():
        os.makedirs(save_location.parent, exist_ok=True)

    if not save_location.exists():
        print('Downloading shiba state dict to', save_location)
        with open(save_location, 'wb') as state_dict_file:
            response = urllib.request.urlopen(download_url)
            data = response.read()
            state_dict_file.write(data)
        print('Done')

    return torch.load(save_location, map_location=torch.device('cpu'))


class ShibaForTask(torch.nn.Module):
    def __init__(self, **kwargs):
        super(ShibaForTask, self).__init__()
        self.shiba_model = Shiba(**kwargs)

    def load_encoder_checkpoint(self, checkpoint_location: Optional[str] = None):
        if checkpoint_location is None:
            state_dict = get_pretrained_state_dict()
        else:
            state_dict = torch.load(checkpoint_location, map_location=torch.device('cpu'))

        self.shiba_model.load_state_dict(state_dict)


class ShibaForSequenceLabeling(ShibaForTask):
    def __init__(self, vocab_size: int, **kwargs):
        super(ShibaForSequenceLabeling, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.config = self.shiba_model.config
        self.config.vocab_size = self.vocab_size
        self.label_layer = torch.nn.Linear(self.shiba_model.config.hidden_size, self.vocab_size)
        self.dropout = torch.nn.Dropout(p=self.shiba_model.config.dropout)

        self.log_softmax = torch.nn.LogSoftmax(dim=2)
        self.loss = torch.nn.NLLLoss()

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor],
                attention_mask: torch.Tensor) -> Tuple:
        embeddings = self.shiba_model(input_ids, attention_mask, None)['embeddings']

        label_hidden_states = self.label_layer(self.dropout(embeddings))
        label_probs = self.log_softmax(label_hidden_states)

        output = {
            'embeddings': embeddings,
            'label_probs': label_probs
        }

        if labels is not None:
            output['loss'] = self.loss(label_probs.transpose(1, 2), labels)

        return output.get('loss', None), output['label_probs'], output['embeddings']


class ShibaForClassification(ShibaForTask):
    def __init__(self, vocab_size: int, **kwargs):
        super(ShibaForClassification, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.config = self.shiba_model.config
        self.config.vocab_size = self.vocab_size
        self.label_layer = torch.nn.Linear(self.shiba_model.config.hidden_size, self.vocab_size)
        self.dropout = torch.nn.Dropout(p=self.shiba_model.config.dropout)

        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.loss = torch.nn.NLLLoss()

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor],
                attention_mask: torch.Tensor) -> Tuple:
        cls_embeddings = self.shiba_model(input_ids, attention_mask, None)['embeddings'][:, 0, :]
        class_hidden_states = self.label_layer(self.dropout(cls_embeddings))
        class_probs = self.log_softmax(class_hidden_states)

        output = {
            'cls_embeddings': cls_embeddings,
            'class_probs': class_probs
        }

        if labels is not None:
            output['loss'] = self.loss(class_probs, labels)

        return output.get('loss', None), output['class_probs'], output['cls_embeddings']


class ShibaForMaskedLanguageModeling(ShibaForTask):
    def __init__(self, vocab_size: int, **kwargs):
        """If vocab size is < than special token codepoints (which it likely is, at least for Japanese), the model will
        be unable to predict special tokens. However, the model shouldn't be trained to predict special tokens anyway"""
        super(ShibaForMaskedLanguageModeling, self).__init__(**kwargs)
        self.vocab_size = vocab_size + 1
        self.unk_token = vocab_size  # we hash the input so there are unknown tokens, but our output vocab is limited
        self.lm_layer = torch.nn.Linear(self.shiba_model.config.hidden_size, self.vocab_size)
        self.config = self.shiba_model.config
        self.config.vocab_size = self.vocab_size

        self.log_softmax = torch.nn.LogSoftmax(dim=2)
        self.loss = torch.nn.NLLLoss(reduction="none")  # https://github.com/microsoft/DeepSpeed/issues/962

    def _replace_unkown_tokens(self, labels: torch.Tensor) -> torch.Tensor:
        return labels.where(labels < self.vocab_size, torch.full(labels.shape, self.unk_token,
                                                                 device=labels.device).long())

    def _compute_loss(self, embeddings: torch.Tensor, char_probs: torch.Tensor, predict_indices: torch.Tensor,
                      labels: Optional[torch.Tensor]) -> Tuple:
        output = {
            'embeddings': embeddings,
            'char_probs': char_probs
        }

        if labels is not None:
            prediction_target_ids = self._replace_unkown_tokens(labels.gather(1, predict_indices))
            loss = self.loss(char_probs.transpose(1, 2), prediction_target_ids).mean() # https://github.com/microsoft/DeepSpeed/issues/962
            output['loss'] = loss

        return output.get('loss', None), output['char_probs'], output['embeddings']

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor],
                attention_mask: torch.Tensor,
                predict_indices: torch.Tensor) -> Tuple:
        output_for_predictions = self.shiba_model(input_ids, attention_mask, predict_indices)['embeddings']

        lm_hidden_states = self.lm_layer(output_for_predictions)
        char_probs = self.log_softmax(lm_hidden_states)

        return self._compute_loss(output_for_predictions, char_probs, predict_indices, labels)


class ShibaForAutoregressiveLanguageModeling(ShibaForMaskedLanguageModeling):
    def _get_causal_mask(self, output_for_predictions: torch.Tensor) -> torch.Tensor:
        causal_mask = (torch.triu(torch.ones(output_for_predictions.shape[1],
                                             output_for_predictions.shape[1])) == 1).transpose(0, 1)
        causal_mask = causal_mask.float().masked_fill(causal_mask == 0, float('-inf')).masked_fill(causal_mask == 1,
                                                                                                   float(0.0))
        return causal_mask.to(output_for_predictions.device)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor],
                attention_mask: torch.Tensor,
                predict_indices: torch.Tensor) -> Tuple:
        output_for_predictions = self.shiba_model(input_ids, attention_mask, predict_indices)['embeddings']

        causal_mask = self._get_causal_mask(output_for_predictions)

        autoregressive_char_seq = self.autregressive_encoder(output_for_predictions.transpose(0, 1),
                                                             src_mask=causal_mask).transpose(0, 1)

        lm_hidden_states = self.lm_layer(autoregressive_char_seq)
        char_probs = self.log_softmax(lm_hidden_states)

        return self._compute_loss(output_for_predictions, char_probs, predict_indices, labels)

    def __init__(self, vocab_size: int, **kwargs):
        super(ShibaForAutoregressiveLanguageModeling, self).__init__(vocab_size=vocab_size, **kwargs)

        self.autregressive_encoder = torch.nn.TransformerEncoderLayer(self.shiba_model.config.hidden_size,
                                                                      self.shiba_model.config.attention_heads,
                                                                      dim_feedforward=self.shiba_model.config.transformer_ff_size,
                                                                      dropout=self.shiba_model.config.dropout,
                                                                      activation=self.shiba_model.config.activation)