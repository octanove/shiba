from typing import Optional

import torch
from local_attention import LocalAttention


# modeled after https://github.com/allenai/allennlp/blob/main/allennlp/modules/transformer/self_attention.py
class LocalSelfAttention(torch.nn.Module):
    """
    expects input of shape [batch_size, sequence_length, embedding_dim]
    """
    def __init__(self, hidden_size: int, num_attention_heads: int, window_size: int = 128, dropout: float = 0.1,
                 bias: bool = True, autopad: bool = True):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )

        if window_size % 2 != 0:
            raise ValueError('Window size must be an even number so it can be split evenly across previous and'
                             'following tokens.')
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = torch.nn.Linear(hidden_size, self.all_head_size, bias=bias)
        self.key = torch.nn.Linear(hidden_size, self.all_head_size, bias=bias)
        self.value = torch.nn.Linear(hidden_size, self.all_head_size, bias=bias)

        # includes softmax and dropout, so we don't need to declare those here
        self.attn = LocalAttention(
                dim=self.attention_head_size,  # setting dim causes positional embeddings to be used
                # LocalAttention looks forward and backward the full window_size value, so by cutting it in half the
                # total is our desired window size. see: https://github.com/lucidrains/local-attention/issues/2
                window_size=window_size // 2,
                causal=False,
                look_forward=1,
                look_backward=1,
                dropout=dropout,
                autopad=autopad
            )

    def forward(
            self,
            input_states: torch.Tensor,
            attention_mask: Optional[torch.BoolTensor] = None):

        mixed_query_layer = self.query(input_states)
        mixed_key_layer = self.key(input_states)
        mixed_value_layer = self.value(input_states)

        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        # input should be of shape (batch_size, attention_head_count, seq_length, attention_head_size)
        context_layer = self.attn(query_layer, key_layer, value_layer, attention_mask)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # it would be nice if we could also optionally output attention probabilities the way allennlp does, but
        # that would require changes to the local_attention package

        return context_layer

    def _transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
