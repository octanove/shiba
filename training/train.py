import os

import torch
import transformers
from transformers import HfArgumentParser, Trainer

from helpers import MAX_JP_CODEPOINT, DataArguments, prepare_data, \
    ShibaTrainingArguments, get_model_hyperparams
from masking import RandomSpanMaskingDataCollator, RandomMaskingDataCollator
from shiba import ShibaForAutoregressiveLanguageModeling, CodepointTokenizer


def main():
    transformers.logging.set_verbosity_info()
    parser = HfArgumentParser((DataArguments, ShibaTrainingArguments))

    data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = CodepointTokenizer()
    if training_args.masking_type == 'bpe_span':
        print('BPE based-span masking')
        data_collator = RandomSpanMaskingDataCollator(tokenizer, True)
    elif training_args.masking_type == 'rand_span':
        print('Random span masking')
        data_collator = RandomSpanMaskingDataCollator(tokenizer, False)
    elif training_args.masking_type == 'rand_char':
        print('Random character masking')
        # char range: https://stackoverflow.com/a/30200250/4243650
        # we aren't including half width stuff
        data_collator = RandomMaskingDataCollator(tokenizer, range(3000, MAX_JP_CODEPOINT))
    else:
        raise RuntimeError('Unknown masking type')

    training_args.logging_dir = training_args.output_dir
    training_data, dev_data = prepare_data(data_args)
    model_hyperparams = get_model_hyperparams(training_args)

    model = ShibaForAutoregressiveLanguageModeling(MAX_JP_CODEPOINT, **model_hyperparams)

    checkpoint_dir = None
    if training_args.resume_from_checkpoint:
        if training_args.load_only_model:
            model.load_state_dict(torch.load(training_args.resume_from_checkpoint))
        else:
            checkpoint_dir = training_args.resume_from_checkpoint
    os.environ['WANDB_PROJECT'] = 'shiba'

    print(training_args)
    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=training_data,
                      eval_dataset=dev_data,
                      )

    trainer.train(resume_from_checkpoint=checkpoint_dir)


if __name__ == '__main__':
    main()
