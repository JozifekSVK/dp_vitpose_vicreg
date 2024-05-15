# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

### DP - this code is modified main_vicreg from from original paper
### We added here more logs, loading model from timm, using MS COCO dataset, etc.
from pathlib import Path
import argparse
import json
import math
import os
import time
from datetime import datetime

import timm
import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
from tqdm import tqdm
import augmentations as aug

from statistics import mean, stdev
from ms_coco import CocoDetection

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)

    # Data
    parser.add_argument("--data-dir", type=Path, default="/path/to/imagenet", required=True,
                        help='Path to the image net dataset')

    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="/content/experiment_folder/",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=60,
                        help='Print logs to the stats.txt file every [log-freq-time] seconds')

    # Model
    parser.add_argument("--arch", type=str, default="resnet50",
                        help='Architecture of the backbone encoder network')
    parser.add_argument("--mlp", default="8192-8192-8192",
                        help='Size and number of layers of the MLP expander head')
    parser.add_argument("--no-projector", default="False",
                        help='Flag if projector will be used')

    # Optim
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=2048,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float, default=0.2,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')

    # Loss
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')

    # Running
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--gpu', type=int, default=0,
                        help='device ID to use for training / testing')

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    torch.backends.cudnn.benchmark = True
    print(args)
    gpu = torch.device(args.device, args.gpu)

    transforms = aug.TrainTransform()

    dataset = CocoDetection(str(args.data_dir) + '/train2017/', str(args.data_dir) + '/annotations/person_keypoints_train2017.json', transform=transforms)

    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    model = VICReg(args).cuda(gpu)

    optimizer = torch.optim.Adam(
      model.parameters(), 
      lr=args.base_lr,
      weight_decay=args.wd
    )

    pathname = os.path.abspath(args.exp_dir)
    current_dateTime = str(datetime.now()).split('.')[0].replace(' ', '-')
    
    if args.no_projector == "True":
      directory_name = f"{pathname}/{args.base_lr}_no_projector_{current_dateTime}"
    else:
      directory_name = f"{pathname}/{args.base_lr}_{args.mlp}_patch_vicreg_{current_dateTime}"
    
    os.mkdir(directory_name)
    stats_file = open(f"{directory_name}/stats.txt", "a", buffering=1)

    start_epoch = 0

    ### Getting model size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))

    start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch - {epoch}")
        for value in enumerate(tqdm(loader)):
            step = value[0]
            x = value[1][0]
            y = value[1][1]
            x = x.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():

                loss = model.forward(x, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_time = time.time()
            if current_time - last_logging > args.log_freq_time:
                stats = dict(
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    time=int(current_time - start_time),
                    lr=args.base_lr,
                )
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
                last_logging = current_time

        ### Calculating validation L2-norm ###
        model.eval()
        res = []
        for value in enumerate(tqdm(loader)):
          step = value[0]
          x = value[1][0]
          y = value[1][1]
          
          if step == 10:
            break
          
          x = x.cuda(gpu, non_blocking=True)
          x_ = model.backbone( x ).to('cpu')
          x_ = torch.flatten(x_, start_dim=1)

          norm_base = x_.norm(dim=1, p=2).tolist()
          res += norm_base

        stats = dict(
            mean=mean(res),
            std=stdev(res)
        )
        print(json.dumps(stats))
        print(json.dumps(stats), file=stats_file)

        model.train()

        ### Saving model
        state = dict(
            epoch=epoch + 1,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
        torch.save(state, directory_name + "/" + "model.pth")
        torch.save(model.backbone.state_dict(), directory_name + "/" + "backbone_trained.pth")

    torch.save(model.backbone.state_dict(), directory_name + "/" + "backbone_trained.pth")


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 100
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = 384
        self.backbone = timm.create_model(
            'vit_small_patch16_224',
            pretrained=False,
            num_classes=0,  # remove classifier nn.Linear
            class_token=False,
            global_pool='avg'
        )

        if args.no_projector == "True":
          self.num_features = 2024
        else:
          self.num_features = int(args.mlp.split("-")[-1])

        self.projector = Projector(args, self.embedding)

    def forward(self, x, y):
        x_ = self.backbone.forward_features(x)
        y_ = self.backbone.forward_features(y)

        if args.no_projector == "True":
          
          x = x_
          y = y_
        else:
          
          x_ = torch.reshape(x_, (-1, self.embedding))
          y_ = torch.reshape(y_, (-1, self.embedding))
          
          x = self.projector(x_)
          y = self.projector(y_)

        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
                self.args.sim_coeff * repr_loss
                + self.args.std_coeff * std_loss
                + self.args.cov_coeff * cov_loss
        )
        return loss


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class LARS(optim.Optimizer):
    def __init__(
            self,
            params,
            lr,
            weight_decay=0,
            momentum=0.9,
            eta=0.001,
            weight_decay_filter=None,
            lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser('VICReg training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
