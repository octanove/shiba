import inspect
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

import torch.nn
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset, load_dataset
from transformers import TrainingArguments

from shiba import Shiba, CodepointTokenizer

MAX_JP_CODEPOINT = 0x9faf
EVAL_DATA_PERCENT = 0.02

@dataclass
class DataArguments:
    data: str = field(
        default=None, metadata={"help": "The location of the Japanese wiki data to use for training."}
    )


@dataclass
class ShibaTrainingArguments(TrainingArguments):
    masking_type: Optional[str] = field(default='rand_span')
    load_only_model: Optional[bool] = field(default=False)

    group_by_length: Optional[bool] = field(default=True)
    logging_first_step: Optional[bool] = field(default=True)
    learning_rate: Optional[float] = 0.001

    logging_steps: Optional[int] = field(default=200)
    report_to: Optional[List[str]] = field(default_factory=lambda: ['wandb'])
    evaluation_strategy: Optional[str] = field(default='steps')
    fp16: Optional[bool] = field(default=torch.cuda.is_available())
    deepspeed: Optional = field(default=None)
    warmup_ratio: Optional[float] = 0.025  # from canine

    per_device_eval_batch_size: Optional[int] = field(default=12)
    per_device_train_batch_size: Optional[int] = field(default=12)
    # max that we can fit on one GPU is 12. 12 * 21 * 8 = 2016
    gradient_accumulation_steps: Optional[int] = field(default=21)

    # model arguments - these have to be in training args for the hyperparam search
    dropout: Optional[float] = field(
        default=0.1
    )
    deep_transformer_stack_layers: Optional[int] = field(
        default=12
    )
    local_attention_window: Optional[int] = field(default=128)


@dataclass
class ShibaWordSegArgs(ShibaTrainingArguments):
    do_predict: Optional[bool] = field(default=True)

    # only used for hyperparameter search
    trials: Optional[int] = field(default=2)
    deepspeed: Optional = field(default=None)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    report_to: Optional[List[str]] = field(default=lambda: ['tensorboard', 'wandb'])
    num_train_epochs: Optional[int] = 6
    save_strategy: Optional[str] = 'no'

    pretrained_bert: Optional[str] = field(default=None)


@dataclass
class ShibaClassificationArgs(ShibaTrainingArguments):
    do_predict: Optional[bool] = field(default=True)
    eval_steps: Optional[int] = field(default=300)
    logging_steps: Optional[int] = field(default=100)
    learning_rate: Optional[float] = 2e-5
    per_device_train_batch_size: Optional[int] = 6
    num_train_epochs: Optional[int] = 6
    save_strategy: Optional[str] = 'no'

    # only used for hyperparameter search
    trials: Optional[int] = field(default=2)
    deepspeed: Optional = field(default=None)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    report_to: Optional[List[str]] = field(default=lambda: ['tensorboard', 'wandb'])

    pretrained_bert: Optional[str] = field(default=None)


def get_model_hyperparams(input_args):
    if not isinstance(input_args, dict):
        input_args = input_args.__dict__

    shiba_hyperparams = inspect.getfullargspec(Shiba.__init__).args
    return {key: val for key, val in input_args.items() if key in shiba_hyperparams}


def get_base_shiba_state_dict(state_dict: Dict) -> Dict:
    if sum(1 for x in state_dict.keys() if x.startswith('shiba_model')) > 0:
        return {key[12:]: val for key, val in state_dict.items() if key.startswith('shiba_model')}
    else:
        return state_dict


def prepare_data(args: DataArguments) -> Tuple[Dataset, Dataset]:
    all_data = load_dataset('json', data_files=args.data)['train']
    data_dict = all_data.train_test_split(train_size=0.98, seed=42)
    training_data = data_dict['train']
    dev_data = data_dict['test']
    return training_data, dev_data


class SequenceLabelingDataCollator:
    def __init__(self):
        self.tokenizer = CodepointTokenizer()

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        padded_batch = self.tokenizer.pad([x['input_ids'] for x in batch])
        input_ids = padded_batch['input_ids']
        attention_mask = padded_batch['attention_mask']

        # don't compute loss from padding
        labels = pad_sequence([torch.tensor(x['labels']) for x in batch], batch_first=True, padding_value=-100)
        # also don't compute loss from CLS or SEP tokens
        special_token_mask = (input_ids == self.tokenizer.CLS) | (input_ids == self.tokenizer.SEP)
        labels = labels.where(~special_token_mask, torch.full(labels.shape, -100))

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class ClassificationDataCollator:
    def __init__(self):
        self.tokenizer = CodepointTokenizer()

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        padded_batch = self.tokenizer.pad([x['input_ids'] for x in batch])
        input_ids = padded_batch['input_ids']
        attention_mask = padded_batch['attention_mask']

        labels = torch.tensor([x['labels'] for x in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
