import os
import tqdm
import argparse
from omegaconf import OmegaConf

import torch
import accelerate
import torch.nn as nn
from torchvision.utils import save_image

from utils.logger import get_logger
from utils.misc import instantiate_from_config, amortize


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to training configuration file',
    )
    parser.add_argument(
        '--seed', type=int, default=8888,
        help='Set random seed',
    )
    parser.add_argument(
        '--mode', type=str, default='sample',
        choices=['sample'],
        help='Choose a sample mode',
    )
    parser.add_argument(
        '--repeat', type=int, default=1,
        help='Number of times to applying the model repeatedly',
    )
    parser.add_argument(
        '--weights', type=str, required=True,
        help='Path to pretrained model weights',
    )
    parser.add_argument(
        '--n_samples', type=int, required=True,
        help='Number of samples',
    )
    parser.add_argument(
        '--save_dir', type=str, required=True,
        help='Path to directory saving samples',
    )
    parser.add_argument(
        '--batch_size', type=int, default=500,
        help='Batch size during sampling',
    )
    return parser


def main():
    # PARSE ARGS AND CONFIGS
    args, unknown_args = get_parser().parse_known_args()
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    unknown_args = [f'{k}={v}' for k, v in zip(unknown_args[::2], unknown_args[1::2])]
    conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(unknown_args))

    # INITIALIZE ACCELERATOR
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}')
    accelerator.wait_for_everyone()

    # INITIALIZE LOGGER
    logger = get_logger(
        use_tqdm_handler=True,
        is_main_process=accelerator.is_main_process,
    )

    # SET SEED
    accelerate.utils.set_seed(args.seed, device_specific=True)
    logger.info('=' * 19 + ' System Info ' + '=' * 18)
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')

    accelerator.wait_for_everyone()

    # BUILD MODELS
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    model: nn.Module = instantiate_from_config(conf.model).eval().to(device)

    stats = None
    if conf.train.noise_type == 'fft':
        stats = torch.load(conf.train.noise_stats)

    # LOAD WEIGHTS
    ckpt = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load model from {args.weights}')
    logger.info('=' * 50)

    accelerator.wait_for_everyone()

    def get_noise(shape):
        if conf.train.noise_type == 'gaussian':
            z = torch.randn(*shape)
        elif conf.train.noise_type == 'fft':
            fft_z = torch.complex(
                real=stats['real_mean'] + stats['real_std'] * torch.randn(*shape),
                imag=stats['imag_mean'] + stats['imag_std'] * torch.randn(*shape),
            )
            z = torch.fft.ifft2(fft_z).real
        else:
            raise ValueError(f'Unknown noise type {conf.train.noise_type}')
        return z.to(device)

    @accelerator.on_main_process
    @torch.no_grad()
    def sample():
        idx = 0
        os.makedirs(args.save_dir, exist_ok=True)
        folds = amortize(args.n_samples, args.batch_size)
        for bs in tqdm.tqdm(folds):
            img_shape = (conf.model.params.img_channels, conf.data.params.img_size, conf.data.params.img_size)
            samples = get_noise((bs, *img_shape))
            for _ in range(args.repeat):
                samples = model(samples)
            if args.repeat == 0:  # noise is out of range [-1, 1]
                samples = (samples - samples.min()) / (samples.max() - samples.min())   # [0, 1]
                samples = samples * 2 - 1                                               # [-1, 1]
            samples = samples.cpu()
            for x in samples:
                save_image(
                    x, os.path.join(args.save_dir, f'{idx}.png'),
                    nrow=1, normalize=True, value_range=(-1, 1),
                )
                idx += 1

    # START SAMPLING
    logger.info('Start sampling...')
    if args.mode == 'sample':
        sample()
    else:
        raise ValueError(f'Unknown sample mode: {args.mode}')
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')


if __name__ == '__main__':
    main()
