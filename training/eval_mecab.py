from typing import List, Dict

import torch
import torchmetrics
from datasets import load_dataset, Dataset


import fugashi


def process_example(example: Dict) -> Dict:
    tokens = example['tokens']
    text = ''.join(tokens)
    labels = []
    for token in tokens:
        new_label_count = len(token)
        new_labels = [1] + [0] * (new_label_count - 1)
        labels.extend(new_labels)

    return {
        'text': text,
        'labels': labels
    }


def mecab_split_to_labels(wakagachi_string: str) -> List[int]:
    labels = [1]
    prev_line_space = False
    for char in wakagachi_string[1:]:
        if char == ' ':
            labels.append(1)
            prev_line_space = True
        else:
            if prev_line_space:
                prev_line_space = False
            else:
                labels.append(0)

    return labels


def compute_score(dataset: Dataset, mecab: fugashi.Tagger) -> float:
    metric = torchmetrics.F1(multiclass=False)

    for row in dataset:
        predicted_labels = mecab_split_to_labels(mecab.parse(row['text']))
        actual_labels = row['labels']
        row_labels = torch.tensor(actual_labels)
        row_predictions = torch.tensor(predicted_labels)
        assert row_labels.shape == row_predictions.shape
        metric.update(row_predictions, row_labels)

    return metric.compute().item()


def main():
    dep = load_dataset('universal_dependencies', 'ja_gsd')
    dep = dep.map(process_example, remove_columns=list(dep['train'][0].keys()))
    dev_data = dep['validation']
    test_data = dep['test']

    mecab = fugashi.Tagger('-Owakati')

    dev_score = compute_score(dev_data, mecab)
    test_score = compute_score(test_data, mecab)

    print('dev f1', dev_score)
    print('test f1', test_score)


if __name__ == '__main__':
    main()
