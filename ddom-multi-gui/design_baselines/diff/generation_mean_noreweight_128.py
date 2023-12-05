import json
import os
import random
import string
import uuid
import shutil

from typing import Optional, Union
from pprint import pprint

import configargparse

import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import csv


@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


with suppress_output():
    import design_bench

    from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
    from design_bench.datasets.discrete.tf_bind_10_dataset import TFBind10Dataset
    from design_bench.datasets.discrete.cifar_nas_dataset import CIFARNASDataset
    from design_bench.datasets.discrete.chembl_dataset import ChEMBLDataset

    from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
    from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset
    from design_bench.datasets.continuous.superconductor_dataset import SuperconductorDataset
    # from design_bench.datasets.continuous.hopper_controller_dataset import HopperControllerDataset

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import pickle as pkl

import torch
from torch.utils.data import Dataset, DataLoader

from nets import DiffusionTest, DiffusionScore
from util import TASKNAME2TASK, configure_gpu, set_seed, get_weights
from forward import ForwardModel
from classifier_model import Classifier

args_filename = "args.json"
checkpoint_dir = "checkpoints"
wandb_project = "sde-flow"
import matplotlib.pyplot as plt



@torch.no_grad()
def run_evaluate(
    taskname,
    seed,
    hidden_size,
    learning_rate,
    checkpoint_path,
    args,
    wandb_logger=None,
    device=None,
    normalise_x=False,
    normalise_y=False,
):
    set_seed(seed)
    task = design_bench.make(TASKNAME2TASK[taskname])

    if task.is_discrete:
        task.map_to_logits()

    if normalise_x:
        task.map_normalize_x()
    if normalise_y:
        task.map_normalize_y()

    # if task.is_discrete:
    #     task.map_to_logits()

    if not args.score_matching:
        model = DiffusionTest.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            taskname=taskname,
            task=task,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            vtype=args.vtype,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            T0=args.T0,
            dropout_p=args.dropout_p)
    else:
        print("Score matching loss")
        model = DiffusionScore.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            taskname=taskname,
            task=task,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            vtype=args.vtype,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            T0=args.T0,
            dropout_p=args.dropout_p)

    model = model.to(device)
    model.eval()

    def _get_classifier():
        checkpoint_path = f"experiments/{taskname}/new_classifier/123_128/wandb/latest-run/files/checkpoints/last.ckpt"
        # checkpoint_path = f"experiments/{taskname}/branin_1_1/123/wandb/latest-run/files/checkpoints/last.ckpt"
        classifier = Classifier.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            taskname=taskname,
            task=task,)
        return classifier
    

    def cond_fn(x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            y_pred = classifier.mlp(x_in, t)  # y_pred represents the continuous predicted values
            # print(x_in.shape, y_pred.shape)
            # print(x_in)
            # print(y_pred)
            loss = (y_pred.view(-1) - y.view(-1))**2  # compute MSE loss
            grad_y_pred = torch.autograd.grad(y_pred.sum(), x_in)[0]
            # return torch.autograd.grad(loss.sum(), x_in)[0] * 1.0
            return grad_y_pred

    def heun_sampler(sde, x_0, ya, num_steps, lmbd=0., keep_all_samples=True):
        device = sde.gen_sde.T.device
        batch_size = x_0.size(0)
        ndim = x_0.dim() - 1
        T_ = sde.gen_sde.T.cpu().item()
        delta = T_ / num_steps
        ts = torch.linspace(0, 1, num_steps + 1) * T_

        # sample
        xs = []
        x_t = x_0.detach().clone().to(device)
        t = torch.zeros(batch_size, *([1] * ndim), device=device)
        t_n = torch.zeros(batch_size, *([1] * ndim), device=device)
        with torch.no_grad():
            for i in range(num_steps):
                t.fill_(ts[i].item())
                if i < num_steps - 1:
                    t_n.fill_(ts[i + 1].item())
                # mu = sde.gen_sde.mu(t, x_t, ya, lmbd=lmbd, gamma=args.gamma)
                mu = sde.gen_sde.mu(t, x_t, lmbd=lmbd, gamma=args.gamma)
                sigma = sde.gen_sde.sigma(t, x_t, lmbd=lmbd)

                # Add gradient
                gradient = cond_fn(x_t, t, ya)
                mu = (
                    # mu.float() + sigma * gradient.float()
                    mu.float() + args.coefficient *gradient.float()
                    # mu.float() + args.coefficient * (i+1) / num_steps * gradient.float()
                )

                x_t = x_t + delta * mu + delta**0.5 * sigma * torch.randn_like(
                    x_t
                )  # one step update of Euler Maruyama method with a step size delta
                # Additional terms for Heun's method
                if i < num_steps - 1:
                    mu2 = sde.gen_sde.mu(t_n,
                                         x_t,
                                        #  ya,
                                         lmbd=lmbd,
                                         gamma=args.gamma)
                    sigma2 = sde.gen_sde.sigma(t_n, x_t, lmbd=lmbd)
                    x_t = x_t + (sigma2 -
                                 sigma) / 2 * delta**0.5 * torch.randn_like(x_t)

                if keep_all_samples or i == num_steps - 1:
                    xs.append(x_t.cpu())
                else:
                    pass
        return xs

    def euler_maruyama_sampler(sde,
                               x_0,
                               ya,
                               num_steps,
                               lmbd=0.,
                               keep_all_samples=True):
        """
        Euler Maruyama method with a step size delta
        """
        # init
        device = sde.gen_sde.T.device
        batch_size = x_0.size(0)
        ndim = x_0.dim() - 1
        T_ = sde.gen_sde.T.cpu().item()
        delta = T_ / num_steps
        ts = torch.linspace(0, 1, num_steps + 1) * T_

        # sample
        xs = []
        x_t = x_0.detach().clone().to(device)
        t = torch.zeros(batch_size, *([1] * ndim), device=device)
        with torch.no_grad():
            for i in range(num_steps):
                t.fill_(ts[i].item())
                # mu = sde.gen_sde.mu(t, x_t, ya, lmbd=lmbd, gamma=args.gamma)
                mu = sde.gen_sde.mu(t, x_t, lmbd=lmbd, gamma=args.gamma)
                sigma = sde.gen_sde.sigma(t, x_t, lmbd=lmbd)

                # Add gradient
                gradient = cond_fn(x_t, t, ya)
                mu = (
                    # mu.float() + sigma * gradient.float()
                    mu.float() + 1*gradient.float()
                )

                x_t = x_t + delta * mu + delta**0.5 * sigma * torch.randn_like(
                    x_t
                )  # one step update of Euler Maruyama method with a step size delta
                if keep_all_samples or i == num_steps - 1:
                    xs.append(x_t.cpu())
                else:
                    pass
        return xs

    num_steps = args.num_steps

    step = []
    value_max = []
    value_mean = []


    for i in range(1, 11, 1):
        step.append(i)
        print("iteration {}".format(i))
        num_samples = 512
        # num_samples = 10

        # lmbds = [0., 1.]
        lmbds = [args.lamda]

        # use the max of the dataset instead
        args.condition = task.y.max()
        # save to file
        expt_save_path = f"./experiments/{args.task}/{args.name}/{args.seed}_noreweight"
        # expt_save_path = f"./experiments/{args.task}/{args.name}/{args.seed}"
        assert os.path.exists(expt_save_path)

        alias = uuid.uuid4()
        run_specific_str = f"{num_samples}_{num_steps}_{args.condition}_{args.gamma}_{args.beta_min}_{args.beta_max}_{args.suffix}_{alias}"
        save_results_dir = os.path.join(
            expt_save_path, f"wandb/latest-run/files/results/{run_specific_str}/")
        if not os.path.exists(save_results_dir):
            os.makedirs(save_results_dir)

        assert os.path.exists(save_results_dir)

        symlink_dir = os.path.join(expt_save_path,
                                f"wandb/latest-run/files/results/latest-run")

        if os.path.exists(symlink_dir):
            os.unlink(symlink_dir)
        os.symlink(run_specific_str, symlink_dir)

        # @torch.no_grad()
        # def _get_trained_model():
        #     checkpoint_path = f"experiments/{taskname}/forward_model/123/wandb/latest-run/files/checkpoints/last.ckpt"
        #     model = ForwardModel.load_from_checkpoint(
        #         checkpoint_path=checkpoint_path,
        #         taskname=taskname,
        #         task=task,)

        #     return model

        # sample and plot
        designs = []
        results = []
        for lmbd in lmbds:
            if not task.is_discrete:
                x_0 = torch.randn(num_samples, task.x.shape[-1],
                                device=device)  # init from prior
            else:
                x_0 = torch.randn(num_samples,
                                task.x.shape[-1] * task.x.shape[-2],
                                device=device)  # init from prior

            y_ = torch.ones(num_samples).to(device) * args.condition
            # xs = euler_maruyama_sampler(model,
            classifier = _get_classifier()
            # classifier.to(x_0.device)
            classifier.to(device)
            xs = heun_sampler(model,
                            x_0,
                            y_,
                            num_steps,
                            lmbd=lmbd,
                            keep_all_samples=False)  # sample
                            # keep_all_samples=True)  # sample

            ctr = 0
            # pred_model = _get_trained_model()
            # preds = []
            for qqq in xs:
                ctr += 1
                print(qqq.shape)
                if not qqq.isnan().any():
                    designs.append(qqq.cpu().numpy())

                    if not task.is_discrete:
                        ys = task.predict(qqq.cpu().numpy())
                    else:
                        qqq = qqq.view(qqq.size(0), -1, task.x.shape[-1])
                        ys = task.predict(qqq.cpu().numpy())
                        print(ys)

                    # pred_ys = pred_model.mlp(qqq)
                    # preds.append(pred_ys.cpu().numpy())

                    max_index = ys.argmax() 
                    print("Corresponding xs for GT ys max: {}".format(qqq[max_index]))
                    print("sum of x entries: {}".format(qqq[max_index].sum().item()) )

                    print("GT ys: {}".format(ys.max()))
                    # print("Pred ys: {}".format(pred_ys.max()))
                    print("Solution: {}".format(task.y.max()))
                    nor_result = (ys.max()-task.y.min())/(task.y.max()-task.y.min())
                    print("normalized output: {}".format(nor_result))
                    if normalise_y:
                        print("normalise")
                        ys = task.denormalize_y(ys)
                        print(ys.max())
                        value_max.append(ys.max().item())
                        # y_mean = task.denormalize_y(ys.mean())
                        value_mean.append(ys.mean().item())
                    else:
                        print("none")
                        print(ys.max())
                    results.append(ys)
                else:
                    print("fuck")

        designs = np.concatenate(designs, axis=0)
        results = np.concatenate(results, axis=0)
        # preds = np.concatenate(preds, axis=0)

        print(designs.shape)
        print(results.shape)
        # print(preds.shape)

        with open(os.path.join(save_results_dir, 'designs.pkl'), 'wb') as f:
            pkl.dump(designs, f)

        with open(os.path.join(save_results_dir, 'results.pkl'), 'wb') as f:
            pkl.dump(results, f)

        # with open(os.path.join(save_results_dir, 'preds.pkl'), 'wb') as f:
        #     pkl.dump(preds, f)

        shutil.copy(args.configs, save_results_dir)
    


    print("step: {}".format(step[1]))
    print("max: {}".format(value_max[1]))
    print("mean: {}".format(value_mean[1]))

    # 使用plot函数绘制第一条折线，它使用x作为x坐标，y1作为y坐标
    plt.plot(step, value_max, label='max', color='blue')

    # 使用plot函数绘制第二条折线，它使用x作为x坐标，y2作为y坐标
    plt.plot(step, value_mean, label='mean', color='red')

    # 为图表添加标题
    plt.title('diffusion process')

    # 为x轴和y轴添加标签
    plt.xlabel('time step')
    plt.ylabel('value')

    # plt.ylim(40,120)

    # 添加图例（这使得图表上显示每条折线的标签）
    plt.legend()

    os.makedirs("diffusion_plot", exist_ok=True)

    # 显示图表
    # plt.savefig(f"diffusion_plot/mean_{args.task}_{args.seed}_{args.coefficient}_noreweight.png")

    print("--------------------------results----------------------------")
    print("mean: {}".format(np.mean(value_max)))
    print("variance: {}".format(np.var(value_max)))

    mean = np.mean(value_max)
    variance = np.var(value_max)
    coe = args.coefficient

    if not os.path.exists('./solution'):
        os.makedirs('./solution')

    filename = f"./solution/{args.task}_noreweight_mean_amend_weigradient_128.csv"
    if not os.path.exists(filename):
        # 如果不存在，创建文件并写入标题行
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([coe, mean, variance])
            
    # 现在，将数据追加到文件
    with open(filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([coe, mean, variance])






if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    # configuration
    parser.add_argument(
        "--configs",
        default=None,
        required=False,
        is_config_file=True,
        help="path(s) to configuration file(s)",
    )
    parser.add_argument('--mode',
                        choices=['train', 'eval'],
                        default='train',
                        required=True)
    parser.add_argument('--task',
                        choices=list(TASKNAME2TASK.keys()),
                        required=True)
    parser.add_argument("--coefficient",
                        type=float,
                        default=0,
                        required=True)
    # reproducibility
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help=
        "sets the random seed; if this is not specified, it is chosen randomly",
    )
    parser.add_argument("--condition", default=0.0, type=float)
    parser.add_argument("--lamda", default=0.0, type=float)
    parser.add_argument("--temp", default='90', type=str)
    parser.add_argument("--suffix", type=str, default="")
    # experiment tracking
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--score_matching", action='store_true', default=False)
    # training
    train_time_group = parser.add_mutually_exclusive_group(required=True)
    train_time_group.add_argument(
        "--epochs",
        default=None,
        type=int,
        help="the number of training epochs.",
    )
    train_time_group.add_argument(
        "--max_steps",
        default=None,
        type=int,
        help=
        "the number of training gradient steps per bootstrap iteration. ignored "
        "if --train_time is set",
    )
    train_time_group.add_argument(
        "--train_time",
        default=None,
        type=str,
        help="how long to train, specified as a DD:HH:MM:SS str",
    )
    parser.add_argument("--num_workers",
                        default=1,
                        type=int,
                        help="Number of workers")
    checkpoint_frequency_group = parser.add_mutually_exclusive_group(
        required=True)
    checkpoint_frequency_group.add_argument(
        "--checkpoint_every_n_epochs",
        default=None,
        type=int,
        help="the period of training epochs for saving checkpoints",
    )
    checkpoint_frequency_group.add_argument(
        "--checkpoint_every_n_steps",
        default=None,
        type=int,
        help="the period of training gradient steps for saving checkpoints",
    )
    checkpoint_frequency_group.add_argument(
        "--checkpoint_time_interval",
        default=None,
        type=str,
        help="how long between saving checkpoints, specified as a HH:MM:SS str",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        required=True,
        help="fraction of data to use for validation",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=False,
        help="place networks and data on the GPU",
    )
    parser.add_argument('--simple_clip', action="store_true", default=False)
    parser.add_argument("--which_gpu",
                        default=0,
                        type=int,
                        help="which GPU to use")
    parser.add_argument(
        "--normalise_x",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--normalise_y",
        action="store_true",
        default=False,
    )

    # i/o
    parser.add_argument('--dataset',
                        type=str,
                        choices=['mnist', 'cifar'],
                        default='mnist')
    parser.add_argument('--dataroot', type=str, default='~/.datasets')
    parser.add_argument('--saveroot', type=str, default='~/.saved')
    parser.add_argument('--expname', type=str, default='default')
    parser.add_argument('--num_steps',
                        type=int,
                        default=1000,
                        help='number of integration steps for sampling')

    # optimization
    parser.add_argument('--T0',
                        type=float,
                        default=1.0,
                        help='integration time')
    parser.add_argument('--vtype',
                        type=str,
                        choices=['rademacher', 'gaussian'],
                        default='rademacher',
                        help='random vector for the Hutchinson trace estimator')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--gamma', type=float, default=1.)

    # model
    parser.add_argument(
        '--real',
        type=eval,
        choices=[True, False],
        default=True,
        help=
        'transforming the data from [0,1] to the real space using the logit function'
    )
    parser.add_argument(
        '--debias',
        action="store_true",
        default=False,
        help=
        'using non-uniform sampling to debias the denoising score matching loss'
    )

    # TODO: remove
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=False,
        help="learning rate for each gradient step",
    )
    parser.add_argument(
        "--auto_tune_lr",
        action="store_true",
        default=False,
        help=
        "have PyTorch Lightning try to automatically find the best learning rate",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        required=False,
        help="size of hidden layers in policy network",
    )
    parser.add_argument(
        "--depth",
        type=int,
        required=False,
        help="number of hidden layers in policy network",
    )
    parser.add_argument(
        "--dropout_p",
        type=float,
        required=False,
        help="dropout probability",
        default=0,
    )
    parser.add_argument(
        "--beta_min",
        type=float,
        required=False,
        default=0.1,
    )
    parser.add_argument(
        "--beta_max",
        type=float,
        required=False,
        default=20.0,
    )
    args = parser.parse_args()

    wandb_project = "score-matching " if args.score_matching else "sde-flow"

    args.seed = np.random.randint(2**31 - 1) if args.seed is None else args.seed
    set_seed(args.seed + 1)
    device = configure_gpu(args.use_gpu, args.which_gpu)

    expt_save_path = f"./experiments/{args.task}/{args.name}/{args.seed}_noreweight"
    # expt_save_path = f"./experiments/{args.task}/{args.name}/{args.seed}"

    if args.mode == 'train':
        if not os.path.exists(expt_save_path):
            os.makedirs(expt_save_path)
        wandb_logger = pl.loggers.wandb.WandbLogger(
            project=wandb_project,
            name=f"{args.name}_task={args.task}_{args.seed}",
            save_dir=expt_save_path)
        log_args(args, wandb_logger)
        run_training(
        # run_training_forward(
            taskname=args.task,
            seed=args.seed,
            wandb_logger=wandb_logger,
            args=args,
            device=device,
        )
    elif args.mode == 'eval':
        checkpoint_path = os.path.join(
            expt_save_path, "wandb/latest-run/files/checkpoints/last.ckpt")
        # checkpoint_path = os.path.join(
        #     expt_save_path, f"wandb/latest-run/files/checkpoints/val.ckpt")
        run_evaluate(taskname=args.task,
                     seed=args.seed,
                     hidden_size=args.hidden_size,
                     args=args,
                     learning_rate=args.learning_rate,
                     checkpoint_path=checkpoint_path,
                     device=device,
                     normalise_x=args.normalise_x,
                     normalise_y=args.normalise_y)
    else:
        raise NotImplementedError
