import os

from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from shiba import CodepointTokenizer
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    save_path = os.path.join(args.output, 'tokenizer.json')

    tokenizer = Tokenizer(BPE(unk_token=chr(CodepointTokenizer.MASK + 1)))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        special_tokens=[chr(x) for x in [CodepointTokenizer.MASK, CodepointTokenizer.SEP, CodepointTokenizer.CLS,
                                         CodepointTokenizer.PAD]])

    tokenizer.train([args.input], trainer)
    tokenizer.save(save_path)
    print('Saved to', save_path)


if __name__ == '__main__':
    main()
