import torch


class PositionEmbedder(torch.nn.Module):

    """
    [batch_size, seq_length, embedding_size]
    """

    def __init__(self, max_sequence_length: int, embedding_dim: int):
        super(PositionEmbedder, self).__init__()
        self.embedding = torch.nn.Embedding(max_sequence_length, embedding_dim, padding_idx=None)

    def forward(self, input_embeddings: torch.Tensor):
        # positional embeddings will broadcast since they're just missing batch dim
        positions = torch.arange(input_embeddings.shape[1], device=input_embeddings.device)
        return input_embeddings + self.embedding(positions)
