import torch

PRIMES = [
    31, 43, 59, 61, 73, 97, 103, 113, 137, 149, 157, 173, 181, 193, 211, 223
]


class MultiHashingEmbedder(torch.nn.Module):

    def __init__(self, embedding_dimension: int = 768, slice_count: int = 8, bucket_count: int = 16000,
                 padding_id: int = 0):
        super().__init__()
        assert embedding_dimension % slice_count == 0, 'Embedding dimension must be divisible by slice count'
        self.embedding_dimension = embedding_dimension
        self.slice_count = slice_count
        self.bucket_count = bucket_count
        self.padding_id = padding_id
        if slice_count > len(PRIMES):
            raise RuntimeError(f"Only support up to {len(PRIMES)} slices")
        self.primes = PRIMES[:slice_count]

        slice_dimension = embedding_dimension // slice_count
        self.buckets = torch.nn.ModuleList([
            torch.nn.Embedding(bucket_count, slice_dimension, padding_idx=padding_id)
            for _ in range(slice_count)
        ])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:

        hashed_ids = [  # use the original ID only for padding
            input_ids.where(input_ids == self.padding_id, (self.primes[k] * input_ids) % self.bucket_count)
            for k in range(self.slice_count)
        ]

        embedding_slices = [self.buckets[k](input_ids) for k, input_ids in enumerate(hashed_ids)]

        return torch.cat(embedding_slices, dim=2)
