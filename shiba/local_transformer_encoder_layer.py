from typing import Optional

import torch

from shiba.local_self_attention import LocalSelfAttention


class LocalTransformerEncoderLayer(torch.nn.Module):
    r"""https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
        except the attention is local. also, expects input of shape [batch_size, sequence_length, embedding_dim]
        whereas the torch implementation switches batch_size and sequence_length

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, window_size=128, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(LocalTransformerEncoderLayer, self).__init__()
        self.self_attn = LocalSelfAttention(d_model, nhead, dropout=dropout, window_size=window_size)
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        activations = {
            'relu': torch.nn.functional.relu,
            'gelu': torch.nn.functional.gelu
        }

        if activation not in activations:
            raise RuntimeError('activation must be in {}, but was {}'.format(set(activations.keys()), activation))

        self.activation = activations[activation]

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(LocalTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, attention_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
