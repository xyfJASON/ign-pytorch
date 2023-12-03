import os
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torch.utils.data import Subset

from utils.misc import instantiate_from_config


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to training configuration file',
    )
    parser.add_argument(
        '-n', '--num', type=int, default=1000,
        help='Number of samples to compute mean and std',
    )
    parser.add_argument(
        '--save_file', type=str, required=True,
        help='Path to save the mean and std',
    )
    return parser


if __name__ == '__main__':
    # ARGS & CONF
    args = get_parser().parse_args()
    conf = OmegaConf.load(args.config)

    # BUILD DATASET
    dataset = instantiate_from_config(conf.data)
    dataset = Subset(dataset, range(args.num))

    # COMPUTE MEAN AND STD
    freqs = []
    for img in tqdm(dataset):  # type: ignore
        img = img[0] if isinstance(img, (tuple, list)) else img
        freqs.append(torch.fft.fft2(img))
    freqs = torch.stack(freqs, dim=0)

    # SAVE MEAN AND STD
    stats = dict(
        real_mean=freqs.real.mean(dim=0, keepdim=True),
        imag_mean=freqs.imag.mean(dim=0, keepdim=True),
        real_std=freqs.real.std(dim=0, keepdim=True),
        imag_std=freqs.imag.std(dim=0, keepdim=True),
    )
    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    torch.save(stats, args.save_file)
    print(f'Statistics saved to {args.save_file}')
