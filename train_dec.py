import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm

# try:
#     import wandb
#
# except ImportError:
#     wandb = None

from model import Generator, Code2Style
from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from non_leaking import augment
from perceptual import LPIPF


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def perceptual_and_l1_loss(real_img, rec_img, real_rep, rec_rep, lambda_l1=0.5):
    return F.mse_loss(200.0*real_rep, 200.0*rec_rep) + lambda_l1 * F.l1_loss(real_img, rec_img)


# def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
#     noise = torch.randn_like(fake_img) / math.sqrt(
#         fake_img.shape[2] * fake_img.shape[3]
#     )
#     grad, = autograd.grad(
#         outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
#     )
#     path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
#
#     path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
#
#     path_penalty = (path_lengths - path_mean).pow(2).mean()
#
#     return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(args, loader, generator, code2style, c2s_optim, device, pf):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        c2s_module = code2style.module

    else:
        g_module = generator
        c2s_module = code2style

    requires_grad(code2style, True)
    requires_grad(generator, False)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        real_img = next(loader)
        real_img = real_img.to(device)

        img_rep = pf(real_img)
        latents = code2style(img_rep)
        rec_img, _ = generator(latents, input_is_latent=True)
        rec_rep = pf(rec_img)

        c2s_loss = perceptual_and_l1_loss(real_img, rec_img, img_rep, rec_rep)
        code2style.zero_grad()
        c2s_loss.backward()
        c2s_optim.step()


        if get_rank() == 0:
            pbar.set_description(
                (
                    f"c2s: {c2s_loss:.4f};"
                )
            )

            if i % 250 == 0:
                utils.save_image(
                    real_img,
                    f"sample/real_{str(i).zfill(6)}.jpg",
                    nrow=4,
                    normalize=True,
                    range=(-1, 1),
                )
                utils.save_image(
                    rec_img,
                    f"sample/rec_{str(i).zfill(6)}.jpg",
                    nrow=4,
                    normalize=True,
                    range=(-1, 1),
                )

            if i % 2000 == 0:
                torch.save(
                    {
                        "c2s": c2s_module.state_dict(),
                        "c2s_optim": c2s_optim.state_dict(),
                        "args": args,
                    },
                    f"checkpoint/c2s_{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    # parser.add_argument(
    #     "--path_regularize",
    #     type=float,
    #     default=2,
    #     help="weight of the path length regularization",
    # )
    # parser.add_argument(
    #     "--path_batch_shrink",
    #     type=int,
    #     default=2,
    #     help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    # )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    # parser.add_argument(
    #     "--g_reg_every",
    #     type=int,
    #     default=4,
    #     help="interval of the applying path length regularization",
    # )
    parser.add_argument(
        "--mixing", type=float, default=0., help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    # parser.add_argument(
    #     "--wandb", action="store_true", help="use weights and biases logging"
    # )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)

    pf = LPIPF(layer_res=[8, 4, 2, 0, 0]).to(device)

    code_dim = 7168
    code2style = Code2Style(args.size, code_dim, args.latent, args.n_mlp).to(device)

    # discriminator = Discriminator(
    #     args.size, channel_multiplier=args.channel_multiplier
    # ).to(device)
    # g_ema = Generator(
    #     args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    # ).to(device)
    # g_ema.eval()
    # accumulate(g_ema, generator, 0)

    # g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    g_reg_ratio = 1.0
    # d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    c2s_optim = optim.Adam(
        code2style.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )

    # g_optim = optim.Adam(
    #     generator.parameters(),
    #     lr=args.lr * g_reg_ratio,
    #     betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    # )
    # d_optim = optim.Adam(
    #     discriminator.parameters(),
    #     lr=args.lr * d_reg_ratio,
    #     betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    # )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"], strict=False)
        # discriminator.load_state_dict(ckpt["d"])
        # g_ema.load_state_dict(ckpt["g_ema"])

        # g_optim.load_state_dict(ckpt["g_optim"])
        # d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        code2style = nn.parallel.DistributedDataParallel(
            code2style,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        # discriminator = nn.parallel.DistributedDataParallel(
        #     discriminator,
        #     device_ids=[args.local_rank],
        #     output_device=args.local_rank,
        #     broadcast_buffers=False,
        # )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # TODO: Check normalization and LPIPF
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    # if get_rank() == 0 and wandb is not None and args.wandb:
    #     wandb.init(project="stylegan 2")

    train(args, loader, generator, code2style, c2s_optim, device, pf)
