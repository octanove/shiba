import os
from typing import Dict

import torch
import torchmetrics
import transformers
from datasets import load_dataset
from transformers import HfArgumentParser, Trainer, EvalPrediction, BertForSequenceClassification, AutoTokenizer, \
    DataCollatorWithPadding

from helpers import DataArguments, get_model_hyperparams, ShibaClassificationArgs, \
    ClassificationDataCollator, get_base_shiba_state_dict
from shiba import ShibaForClassification, CodepointTokenizer


def main():
    transformers.logging.set_verbosity_info()
    parser = HfArgumentParser((ShibaClassificationArgs, DataArguments))

    training_args, data_args = parser.parse_args_into_dataclasses()
    training_args.logging_dir = training_args.output_dir

    all_data = load_dataset('json', data_files=data_args.data)['train']
    categories = {idx: cat_name for idx, cat_name in enumerate({x['category'] for x in all_data})}
    id_by_category = {val: key for key, val in categories.items()}

    if training_args.pretrained_bert is None:
        tokenizer = CodepointTokenizer()
        model_hyperparams = get_model_hyperparams(training_args)
        model = ShibaForClassification(vocab_size=len(categories), **model_hyperparams)
        data_collator = ClassificationDataCollator()

        if training_args.resume_from_checkpoint:
            print('Loading and using base shiba states from', training_args.resume_from_checkpoint)
            checkpoint_state_dict = torch.load(training_args.resume_from_checkpoint)
            model.shiba_model.load_state_dict(get_base_shiba_state_dict(checkpoint_state_dict))

        def process_example(example: Dict) -> Dict:
            return {
                'input_ids': tokenizer.encode(example['text'])['input_ids'][:model.config.max_length],
                'labels': id_by_category[example['category']]
            }

        def compute_metrics(pred: EvalPrediction) -> Dict:
            metric = torchmetrics.Accuracy(num_classes=len(categories))
            label_probs, embeddings = pred.predictions
            labels = torch.tensor(pred.label_ids)
            label_probs = torch.exp(torch.tensor(label_probs))  # undo the log in log softmax, get indices

            metric.update(label_probs, labels)

            accuracy = metric.compute()

            return {
                'accuracy': accuracy.item()
            }

    else:
        model = BertForSequenceClassification.from_pretrained(training_args.pretrained_bert,
                                                              problem_type='single_label_classification',
                                                              num_labels=len(id_by_category))
        tokenizer = AutoTokenizer.from_pretrained(training_args.pretrained_bert)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

        def process_example(example: Dict) -> Dict:
            encoded_example = tokenizer(example['text'], truncation=True)
            encoded_example['label'] = id_by_category[example['category']]
            return encoded_example

        def compute_metrics(pred: EvalPrediction) -> Dict:
            metric = torchmetrics.Accuracy(num_classes=len(categories))
            label_probs = torch.nn.functional.softmax(torch.tensor(pred.predictions))
            labels = torch.tensor(pred.label_ids)

            metric.update(label_probs, labels)

            accuracy = metric.compute()

            return {
                'accuracy': accuracy.item()
            }

    os.environ['WANDB_PROJECT'] = 'shiba'

    dataset_dict = all_data.train_test_split(test_size=0.1, seed=1337)
    dataset_dict = dataset_dict.map(process_example, remove_columns=list(dataset_dict['train'][0].keys()))

    print(training_args)
    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=dataset_dict['train'],
                      eval_dataset=dataset_dict['test'],
                      compute_metrics=compute_metrics
                      )
    trainer.train()


if __name__ == '__main__':
    main()
