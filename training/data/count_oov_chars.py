import argparse
import datasets
from tqdm import tqdm

MAX_JP_CODEPOINT = 0x9faf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', required=True)
    parser.add_argument('--vocabulary_size', type=int, default=MAX_JP_CODEPOINT)

    args = parser.parse_args()

    data = datasets.load_dataset('json', data_files=args.input_data)['train']
    total_chars = 0
    oov_chars = 0

    for example in tqdm(data):
        input_ids = example['input_ids']
        total_chars += len(input_ids)
        oov_chars += sum(1 for i in input_ids if i >= args.vocabulary_size)

    print(f'Found {total_chars} total chars and {oov_chars} out-of-vocab chars (' + '{0:.2f}'.
          format(oov_chars / total_chars) + '%)')


if __name__ == '__main__':
    main()
