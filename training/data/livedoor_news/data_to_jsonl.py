import os

import jsonlines
from glob import glob
from argparse import ArgumentParser
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', required=True, help='Location of the "text" directory from the livedoor ニュースコーパス')
    parser.add_argument('--output', required=True, help='Location to write data to')

    args = parser.parse_args()

    category_folders = glob(f'{args.input}/*/')
    assert len(category_folders) == 9, 'Should be nine categories in the dataset'

    articles = []
    for folder in tqdm(category_folders, desc='Processing categories'):
        category = os.path.split(os.path.split(folder)[0])[-1]
        article_files = glob(f'{folder}/{category}*.txt')
        for article_filename in tqdm(article_files, desc=f'Processing files for category {category}', leave=False):
            with open(article_filename, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
                lines = [x for x in lines if x]
                url, date, title = lines[0:3]
                # not clear if we need the translate, but just to be safe
                # from https://qiita.com/sugulu_Ogawa_ISID/items/697bd03499c1de9cf082
                article_contents = ''.join(lines[3:]).translate(str.maketrans(
                    {'\n': '', '\t': '', '\r': '', '\u3000': ''}))

            articles.append({
                'url': url,
                'date': date,
                'title': title,
                'category': category,
                'text': article_contents
            })

    output_filename = args.output.split('.')[0] + '.jsonl'
    print(f'Writing output data to {output_filename}')
    with jsonlines.open(output_filename, 'w') as writer:
        writer.write_all(tqdm(articles, desc='Writing articles'))

    print('Done')


if __name__ == '__main__':
    main()
