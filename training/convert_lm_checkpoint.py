from argparse import ArgumentParser
import torch
from helpers import get_base_shiba_state_dict


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    lm_state_dict = torch.load(args.input)
    shiba_state_dict = get_base_shiba_state_dict(lm_state_dict)

    torch.save(shiba_state_dict, args.output)


if __name__ == '__main__':
    main()
