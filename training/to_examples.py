import argparse
import jsonlines
from tqdm import tqdm

from shiba import CodepointTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', required=True)
    parser.add_argument('--output_data', required=True)
    parser.add_argument('--max_length', type=int, default=2048,  # from CANINE, but == downsampling_rate*ROBERTA_max_len
                        help='Max length in tokens, including [CLS] and [SEP] tokens')
    parser.add_argument('--max_sents', type=int, default=0)  # from ROBERTA
    parser.add_argument('--cross_docs', type=bool, default=True)  # from ROBERTA

    min_sents = 2
    args = parser.parse_args()

    infile = open(args.input_data)
    outfile = jsonlines.open(args.output_data, 'w')

    current_length = 0
    current_sents = []

    sents_lost_under_min = 0
    sents_lost_cross_doc = 0

    tokenizer = CodepointTokenizer()

    for line in tqdm(infile):
        line_text = line.strip()

        # empty lines represent gaps between docs; if args.cross_docs is true we will append an extra
        # SEP token like ROBERTA does
        if line_text == '' and not args.cross_docs:
            # we hit the end of a document, clear the buffer and keep going
            sents_lost_cross_doc += len(current_sents)
            current_length = 0
            current_sents = []
            continue

        # + len(current_sents) + 1 to account for [CLS] and [SEP] tokens, incl new token that would be added
        if (args.max_sents <= 0 or len(current_sents) < args.max_sents) \
                and len(line_text) + current_length + len(current_sents) + 1 <= args.max_length:
            current_sents.append(line_text)
            current_length += len(line_text)
        else:
            if len(current_sents) >= min_sents:
                encoded_example = tokenizer.encode(current_sents)
                assert len(encoded_example['input_ids']) <= args.max_length

                outfile.write({'input_ids': encoded_example['input_ids'].tolist()})
            else:
                sents_lost_under_min += len(current_sents)

            current_length = 0
            current_sents = []

    print('Finished. Lost', sents_lost_cross_doc, 'sentences due to examples crossing documents and',
          sents_lost_under_min, 'sentences due to examples too long to fit the minimum sentences in an example')

    infile.close()
    outfile.close()


if __name__ == '__main__':
    main()
