import math
import os
import random
from collections import defaultdict
from typing import Tuple, Dict, Set, List
import unicodedata

import torch
from tokenizers import Tokenizer

from shiba import CodepointTokenizer

SPECIAL_TOKENS_WITHOUT_PADDING = {CodepointTokenizer.MASK, CodepointTokenizer.SEP, CodepointTokenizer.CLS}
MAX_SPECIAL_TOKEN = max(SPECIAL_TOKENS_WITHOUT_PADDING)
MIN_SPECIAL_TOKEN = min(SPECIAL_TOKENS_WITHOUT_PADDING)


def _special_tokens_mask(input_ids: torch.Tensor, special_tokens: Set[int]) -> torch.Tensor:
    mask = torch.zeros(input_ids.shape).bool()
    for token_id in special_tokens:
        mask = mask | (input_ids == token_id)

    return mask


def _special_tokens_mask_from_range(input_ids: torch.Tensor, special_tokens: range) -> torch.Tensor:
    return (input_ids >= special_tokens.start) & (input_ids <= special_tokens.stop)


def random_mask(input_ids: torch.Tensor, attention_mask: torch.tensor,
                masking_percent: float = 0.15,
                min_char_replacement: int = 1,
                max_char_replacement: int = CodepointTokenizer.MAX_CODEPOINT) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """The standard way to do this (how HuggingFace does it) is to randomly mask each token with masking_prob. However,
    this can result in a different number of masks for different sentences, which would prevent us from using gather()
    to save compute like CANINE does. consequently, we instead always mask exactly length * masking_prob tokens, and
    instead select those indices randomly. this means that the masked indices can (and likely will) be OUT OF ORDER.
    it also means that in batches with padding, potentially a higher % will be masked in the shorter sentences"""
    labels = input_ids.clone()
    input_ids = input_ids.clone()
    special_tokens_mask = _special_tokens_mask_from_range(input_ids, range(MIN_SPECIAL_TOKEN, MAX_SPECIAL_TOKEN))
    special_tokens_mask = special_tokens_mask | attention_mask.bool()
    mask_count = math.floor(input_ids.shape[1] * masking_percent)

    indices_to_mask = []
    for unmaskable_indices, inputs in zip(special_tokens_mask, input_ids):
        # compute the possible indices we could mask for this input
        maskable_indices = torch.arange(inputs.shape[0]).masked_select(~unmaskable_indices)
        # take mask_count random indices, get the maskable indices using random indices
        indices_to_mask.append(maskable_indices[torch.randperm(maskable_indices.shape[0])[:mask_count]])

    # indices_to_mask is a tensor containing the indices to be masked (I.E. usable with gather()), while
    # masked_indices is a boolean mask indicating whether indices are targets for masking
    indices_to_mask = torch.stack(indices_to_mask)
    masked_indices = torch.full(labels.shape, False).scatter(1, indices_to_mask, torch.full(labels.shape, True))

    # 80% of the time, we replace masked input tokens with mask token
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = CodepointTokenizer.MASK

    # 10% (half of the remaining 20%) of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(low=min_char_replacement, high=max_char_replacement,
                                 size=labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return input_ids, labels, indices_to_mask


def bpe_span_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, replacement_vocab: Dict[int, List[str]],
                  bpe_tokenizer: Tokenizer, masking_percent: float = 0.15):
    # we are masking byte piece spans

    labels = input_ids.clone()
    input_ids = input_ids.clone()
    special_tokens_mask = _special_tokens_mask_from_range(input_ids, range(MIN_SPECIAL_TOKEN, MAX_SPECIAL_TOKEN))
    special_tokens_mask = special_tokens_mask | attention_mask.bool()
    mask_count = math.floor(input_ids.shape[1] * masking_percent)  # total number of tokens to mask

    spans_per_row = []
    all_masked_indices = []
    for unmaskable_indices, inputs in zip(special_tokens_mask, input_ids):
        # compute the possible indices we could mask for this input
        maskable_indices = torch.arange(inputs.shape[0]).masked_select(~unmaskable_indices)
        maskable_indices_set = set(maskable_indices.numpy())

        original_string = unicodedata.normalize("NFKC", ''.join(chr(x) for x in inputs))
        bpe_split_string = bpe_tokenizer.encode(original_string)
        start_length_tuples = [(start, end - start) for start, end in bpe_split_string.offsets
                               if start in maskable_indices_set]
        random.shuffle(start_length_tuples)
        total_masked = 0
        span_iter = iter(start_length_tuples)
        spans_to_mask = []
        while total_masked < mask_count:
            try:
                span_index, span_length = next(span_iter)
                if total_masked + span_length <= mask_count:
                    spans_to_mask.append((span_index, span_length))
                    total_masked += span_length
            except StopIteration:
                print('Warning: randomly masking to fill remaining mask slots')
                candidate_indices = list(maskable_indices_set - {idx for span_idx, span_length in spans_to_mask
                                                                 for idx in range(span_idx, span_idx + span_length)})
                random.shuffle(candidate_indices)
                for idx in candidate_indices:
                    spans_to_mask.append((idx, 1))
                    total_masked += 1
                    if total_masked == mask_count:
                        break

        assert (total_masked == mask_count)
        spans_per_row.append(spans_to_mask)

        all_masked_indices.append(torch.tensor([idx for start_loc, length in spans_to_mask
                                                for idx in range(start_loc, start_loc + length)]))

    span_starts_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor([start_idx for start_idx, length in sublist])
                                                          for sublist in spans_per_row], batch_first=True,
                                                         padding_value=-1)
    span_lengths_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor([length for start_idx, length in sublist])
                                                           for sublist in spans_per_row], batch_first=True,
                                                          padding_value=-1)
    unused_span_indices = span_starts_tensor == -1
    spans_to_replace = torch.zeros(span_starts_tensor.shape).where(unused_span_indices, torch.tensor(0.8))
    spans_to_replace = torch.bernoulli(spans_to_replace).bool()
    spans_to_randomize = torch.bernoulli(torch.full(spans_to_replace.shape,
                                                    0.5)).bool() & ~spans_to_replace & ~unused_span_indices

    for locs, lengths, replace, row_idx in zip(span_starts_tensor, span_lengths_tensor,
                                               spans_to_replace, range(input_ids.shape[0])):
        row_span_start_indices = locs[replace]
        row_span_lengths = lengths[replace]
        span_index_targets = torch.tensor([idx for start_loc, length in zip(row_span_start_indices, row_span_lengths)
                                           for idx in range(start_loc, start_loc + length)], dtype=torch.long)

        if span_index_targets.shape[0] != 0:
            input_ids[row_idx, span_index_targets] = CodepointTokenizer.MASK

    for locs, lengths, randomize, row_idx in zip(span_starts_tensor, span_lengths_tensor,
                                                 spans_to_randomize, range(input_ids.shape[0])):
        row_span_start_indices = locs[randomize]
        row_span_lengths = lengths[randomize]

        # for each span, select a random subword from the byte piece embedding vocab of the same length
        # and use it to replace the target characters
        for start_idx, span_len in zip(row_span_start_indices, row_span_lengths):
            replacement_word = random.choice(replacement_vocab[span_len.item()])
            codepoints = torch.tensor([ord(c) for c in replacement_word], dtype=torch.long)
            input_ids[row_idx, start_idx:start_idx + span_len] = codepoints

    masked_indices = torch.stack(all_masked_indices)
    return input_ids, labels, masked_indices


def random_span_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, replacement_vocab: Dict[int, List[str]],
                     masking_percent: float = 0.15, span_length: int = 2) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """randomly mask spans, and replace some of the spans with same length subwords. note that character-trained canine
    only does masking (no replacement) for some reason, so this is slightly different to what we're doing"""
    labels = input_ids.clone()
    input_ids = input_ids.clone()
    special_tokens_mask = _special_tokens_mask_from_range(input_ids, range(MIN_SPECIAL_TOKEN, MAX_SPECIAL_TOKEN))
    special_tokens_mask = special_tokens_mask | attention_mask.bool()
    mask_count = math.floor(input_ids.shape[1] * masking_percent)  # total number of tokens to mask

    # length of each span; length of this list is total number of spans
    num_spans = math.floor(mask_count / span_length)

    maskable_indices_per_row = []
    span_locations_per_row = []
    masked_indices_per_row = []
    for unmaskable_indices, inputs in zip(special_tokens_mask, input_ids):
        # compute the possible indices we could mask for this input
        maskable_indices = torch.arange(inputs.shape[0]).masked_select(~unmaskable_indices)
        maskable_indices_per_row.append(maskable_indices)

        # this gets a little confusing, but we compute start locations for spans that we later use to index
        # maskable_indices (not the raw indices, because then a span might overlap with an unmaskable token)
        # this does mean a span could conceivably end up being non-contiguous if there's an unmaskable token
        # in the middle of where it should have gone, which is fine
        start_offset = random.randrange(0, span_length)
        row_span_locations = torch.arange(start_offset, maskable_indices.shape[0] - span_length, step=span_length)
        row_span_locations = row_span_locations[torch.randperm(row_span_locations.shape[0])[:num_spans]]
        span_locations_per_row.append(row_span_locations)

        row_masked_indices = torch.cat([row_span_locations] + [row_span_locations+x for x in range(1, span_length)])
        row_masked_indices = maskable_indices[row_masked_indices]
        masked_indices_per_row.append(row_masked_indices[torch.randperm(row_masked_indices.shape[0])])

    all_masked_indices = torch.stack(masked_indices_per_row)

    # locations used to index maskable indices for spans
    span_start_locations = torch.stack(span_locations_per_row)

    # 80% of the time, we replace spans tokens with mask token
    spans_to_replace = torch.bernoulli(torch.full(span_start_locations.shape, 0.8)).bool()

    # 10% (half of the remaining 20%) of the time, we replace span input tokens with random word
    spans_to_randomize = torch.bernoulli(torch.full(span_start_locations.shape, 0.5)).bool() & ~spans_to_replace

    for locs, replace, maskable_indices, row_idx in zip(span_start_locations, spans_to_replace,
                                                        maskable_indices_per_row, range(input_ids.shape[0])):
        if replace.sum() != 0:
            span_starts = locs[replace]  # which of the spans (by start location) should be replaced
            # get the actual target indices looking up all the indices in these spans in maskable_indices
            target_indices = maskable_indices[torch.cat([span_starts] + [span_starts + x for x in range(1, span_length)])]
            input_ids[row_idx, target_indices] = CodepointTokenizer.MASK

            #assert len(set(x.item() for x in all_masked_indices[row_idx]) & set(x.item() for x in target_indices)) == target_indices.shape[0]

    for locs, randomize, maskable_indices, row_idx in zip(span_start_locations, spans_to_randomize,
                                                          maskable_indices_per_row, range(input_ids.shape[0])):

        if randomize.sum() != 0:
            # from span locations get those marked for randomization (replacement), and use those locations to get
            # actual start indices from maskable indices
            span_starts = locs[randomize]  # which of the spans (by start location) should be randomized
            # compute actual spans as a 2d tensor
            spans = torch.stack([span_starts] + [span_starts + x for x in range(1, span_length)]).transpose(0, 1)
            spans = maskable_indices[spans]  # convert them to maskable indices

            # get a random replacement of span length from
            replacements = torch.tensor([[ord(c) for c in random.choice(replacement_vocab[span_length])]
                                        for i in range(spans.shape[0])])

            input_ids[row_idx][spans] = replacements

            #assert len(set(x.item() for x in all_masked_indices[row_idx]) & set(x.item() for x in spans.flatten())) == spans.flatten().shape[0]

    return input_ids, labels, all_masked_indices


class RandomSpanMaskingDataCollator:
    tokenizer_vocab_loc = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bpe_vocab.json')
    special_tokens = SPECIAL_TOKENS_WITHOUT_PADDING | {CodepointTokenizer.PAD}

    def __init__(self, tokenizer: CodepointTokenizer, bpe_span_selection: bool):
        self.tokenizer = tokenizer
        self.jp_tokenizer = Tokenizer.from_file(self.tokenizer_vocab_loc)
        self.jp_vocab = self.jp_tokenizer.get_vocab()
        self.wp_by_length = self._compute_subword_vocab()
        self.bpe_span_selection = bpe_span_selection

    def _compute_subword_vocab(self) -> Dict[int, List[str]]:
        word_pieces = [wp.strip('#') for wp in self.jp_vocab if wp not in self.special_tokens and not wp.isdigit()]
        word_pieces_by_length = defaultdict(list)
        for wp in word_pieces:
            word_pieces_by_length[len(wp)].append(wp)

        return word_pieces_by_length

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        padded_batch = self.tokenizer.pad([x['input_ids'] for x in batch])
        if self.bpe_span_selection:
            input_ids, labels, masked_indices = bpe_span_mask(padded_batch['input_ids'],
                                                              padded_batch['attention_mask'],
                                                              replacement_vocab=self.wp_by_length,
                                                              bpe_tokenizer=self.jp_tokenizer)

        else:
            input_ids, labels, masked_indices = random_span_mask(padded_batch['input_ids'],
                                                                 padded_batch['attention_mask'],
                                                                 replacement_vocab=self.wp_by_length)

        padded_batch.update({
            'input_ids': input_ids,
            'labels': labels,
            'predict_indices': masked_indices
        })

        return padded_batch


class RandomMaskingDataCollator:
    def __init__(self, tokenizer: CodepointTokenizer, replacement_range: range):
        self.tokenizer = tokenizer
        self.replacement_range = replacement_range

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        padded_batch = self.tokenizer.pad([x['input_ids'] for x in batch])

        input_ids, labels, masked_indices = random_mask(padded_batch['input_ids'],
                                                        padded_batch['attention_mask'],
                                                        min_char_replacement=self.replacement_range.start,
                                                        max_char_replacement=self.replacement_range.stop)

        padded_batch.update({
            'input_ids': input_ids,
            'labels': labels,
            'predict_indices': masked_indices
        })

        return padded_batch
