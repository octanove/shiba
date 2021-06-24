import os
from typing import Dict

import torch
import transformers
from datasets import load_dataset
from transformers import HfArgumentParser, Trainer, EvalPrediction, BertForTokenClassification, AutoTokenizer, \
    DataCollatorForTokenClassification

from shiba import CodepointTokenizer, ShibaForSequenceLabeling
from helpers import get_model_hyperparams, SequenceLabelingDataCollator, \
    ShibaWordSegArgs, get_base_shiba_state_dict
import torchmetrics


def main():
    transformers.logging.set_verbosity_info()
    parser = HfArgumentParser((ShibaWordSegArgs,))

    training_args = parser.parse_args_into_dataclasses()[0]
    training_args.logging_dir = training_args.output_dir

    if training_args.pretrained_bert is None:

        tokenizer = CodepointTokenizer()

        def process_example(example: Dict) -> Dict:
            tokens = example['tokens']
            text = ''.join(tokens)
            labels = [0]  # CLS token
            for token in tokens:
                new_label_count = len(token)
                new_labels = [1] + [0] * (new_label_count - 1)
                labels.extend(new_labels)

            input_ids = tokenizer.encode(text)['input_ids']

            return {
                'input_ids': input_ids,
                'labels': labels
            }

        model_hyperparams = get_model_hyperparams(training_args)

        # beginning of word or not beginning of word
        model = ShibaForSequenceLabeling(2, **model_hyperparams)

        if training_args.resume_from_checkpoint:
            print('Loading and using base shiba states from', training_args.resume_from_checkpoint)
            checkpoint_state_dict = torch.load(training_args.resume_from_checkpoint)
            model.shiba_model.load_state_dict(get_base_shiba_state_dict(checkpoint_state_dict))

        data_collator = SequenceLabelingDataCollator()
    else:
        model = BertForTokenClassification.from_pretrained(training_args.pretrained_bert, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(training_args.pretrained_bert)
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

        def process_example(example: Dict) -> Dict:
            tokens = example['tokens']
            text = ''.join(tokens)
            labels = [0]  # CLS token
            for token in tokens:
                new_label_count = len(token)
                new_labels = [1] + [0] * (new_label_count - 1)
                labels.extend(new_labels)

            encoded_example = tokenizer(text, truncation=True)
            encoded_example['labels'] = labels

            return encoded_example

    dep = load_dataset('universal_dependencies', 'ja_gsd')
    dep = dep.map(process_example, remove_columns=list(dep['train'][0].keys()))

    os.environ['WANDB_PROJECT'] = 'shiba'

    def compute_metrics(pred: EvalPrediction) -> Dict:

        if training_args.pretrained_bert is None:
            label_probs, embeddings = pred.predictions
            labels = torch.tensor(pred.label_ids)
            predictions = torch.max(torch.exp(torch.tensor(label_probs)), dim=2)[1]
        else:
            predictions = torch.max(torch.tensor(pred.predictions), dim=2)[1]
            labels = torch.tensor(pred.label_ids)

        metric = torchmetrics.F1(multiclass=False)
        for label_row, prediction_row in zip(labels, predictions):
            row_labels = []
            row_predictions = []
            for lbl, pred in zip(label_row, prediction_row):
                if lbl != -100:
                    row_labels.append(lbl)
                    row_predictions.append(pred)

            row_labels = torch.tensor(row_labels)
            row_predictions = torch.tensor(row_predictions)
            assert row_labels.shape == row_predictions.shape
            metric.update(row_predictions, row_labels)

        f1 = metric.compute()

        return {
            'f1': f1.item()
        }

    print(training_args)

    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=dep['train'],
                      eval_dataset=dep['validation'],
                      compute_metrics=compute_metrics
                      )

    trainer.train()
    posttrain_metrics = trainer.predict(dep['test']).metrics
    print(posttrain_metrics)


if __name__ == '__main__':
    main()
