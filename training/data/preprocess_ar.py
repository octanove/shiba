# preprocessing and cleaning code taken from https://github.com/cl-tohoku/bert-japanese/blob/main/make_corpus_wiki.py
import argparse
import gzip
import json
import re
import unicodedata
# import fugashi
from arabert.preprocess import ArabertPreprocessor
from tqdm import tqdm
import nltk



def filter_text(text):
    # filter out text containing equations
    if "\displaystyle" in text:
        return False

    return True



def main(args):
    # sent_splitter = MeCabSentenceSplitter(args.mecab_option)
    model_name = "aubmindlab/bert-base-arabertv2"
    arabert_prep = ArabertPreprocessor(model_name=model_name)
    
    with open(args.input_file, "r") as input_file, \
         open(args.output_file, "w") as output_file:
        for line in tqdm(input_file):
            if line is None:
                continue

            is_processed = False
            for sentence in nltk.sent_tokenize(line):
                sentence = sentence.strip()
                sentence = arabert_prep.preprocess(sentence)
                if len(sentence) < args.min_text_length:
                    continue
                if len(sentence) > args.max_text_length:
                    continue
                if not filter_text(sentence):
                    continue

#                 assert not "\n" in line
#                 assert sentence != ""
                print(sentence, file=output_file)
                is_processed = True

            if is_processed:
                print("", file=output_file)

                
                
                
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--min_text_length", type=int, default=10)
    parser.add_argument("--max_text_length", type=int, default=1000)
    # parser.add_argument("--mecab_option", type=str)
    args = parser.parse_args()
    main(args)
