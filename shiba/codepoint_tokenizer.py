from typing import List, Dict, Union, Iterable
import torch
from torch.nn.utils.rnn import pad_sequence


class CodepointTokenizer:

    # padding is always 0, which is fine because it's the null unicode point
    # the remaining code points are private use codepoints, see https://en.wikipedia.org/wiki/Private_Use_Areas

    PAD = 0
    CLS = 0xE000
    SEP = 0xE001
    MASK = 0xE003
    _READABLE_SPECIAL_TOKENS = {
         PAD: '[PAD]',
         CLS: '[CLS]',
         SEP: '[SEP]',
         MASK: '[MASK]'
    }

    MAX_CODEPOINT = 0x10FFFF

    def _sequence_to_ids(self, text: Union[str, Iterable[str]]) -> List[int]:

        if isinstance(text, str):
            text = text.strip()
            return [self.CLS] + [ord(c) for c in text]
        else:
            text_iter = iter(text)
            first_text = next(text_iter).strip()
            remaining_texts = [x.strip() for x in text_iter]

            encoded_chars = [self.CLS] + [ord(c) for c in first_text]
            for subsequent_text in remaining_texts:
                encoded_chars += [self.SEP] + [ord(c) for c in subsequent_text]

            return encoded_chars

    def encode(self, inp: Union[str, Iterable[str]]) -> Dict[str, torch.Tensor]:
        return {x: y.flatten() for x, y in self.encode_batch([inp]).items()}

    def encode_batch(self, batch: List[Union[str, Iterable[str]]]) -> Dict[str, torch.Tensor]:
        ids = [
            self._sequence_to_ids(text) for text in batch
        ]
        return self.pad(ids)

    def pad(self, ids_list: List[List[int]]) -> Dict[str, torch.Tensor]:
        id_tensor = pad_sequence([torch.tensor(ids) for ids in ids_list], padding_value=0, batch_first=True)
        attention_mask = (id_tensor == self.PAD)

        return {
            'input_ids': id_tensor,
            'attention_mask': attention_mask
        }

    def decode(self, ids: List[int]) -> str:
        return ''.join(
            self._READABLE_SPECIAL_TOKENS.get(i, chr(i)) for i in ids
        )



