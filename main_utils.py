import torch
import datetime
import uuid
import os
import copy
import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler as _LRScheduler
from baseline_methods import GDModel


def aggregate_metrics(result_dict, d, bootstrap_trials=1000):
    """
    Takes as input a tensor of shape (num_eval, n_points) and returns a dict with
    per-point mean, stddev, and bootstrap limits
    """
    results = {}
    for model_name in result_dict.keys():
        errs = result_dict[model_name]
        tmp = {}
        tmp["mean"] = errs.mean(0) / d
        tmp["std"] = errs.std(0, unbiased=True) / d
        n = len(errs)
        bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
        bootstrap_means = errs[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
        tmp["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :] / d
        tmp["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :] / d
        results[model_name] = tmp

    return results


class CosineAnnealingWarmup(_LRScheduler):

    def __init__(
            self,
            optimizer: Optimizer,
            warmup_steps: int,
            learning_rate: float,
            min_lr: float,
            lr_decay_steps: int,
            verbose: bool = False,
    ):
        self.warmup_steps = warmup_steps  # warm up steps
        self.learning_rate = learning_rate
        self.lr_decay_steps = lr_decay_steps  # beyond this, just return min_lr
        self.min_lr = min_lr
        super().__init__(optimizer=optimizer, last_epoch=-1, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [self.learning_rate * self._step_count / self.warmup_steps
                    for group in self.optimizer.param_groups]
        if self._step_count > self.lr_decay_steps:
            return [self.min_lr for group in self.optimizer.param_groups]

        decay_ratio = (
                (self._step_count - self.warmup_steps)
                / (self.lr_decay_steps - self.warmup_steps)
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return [self.min_lr + coeff * (self.learning_rate - self.min_lr)
                for group in self.optimizer.param_groups]


def get_newton_loss(max_num_step, newton_model_dict, xs, ys):
    # for newton method running stably, we need to use small batch size
    d = xs.shape[-1]
    newton_loss_dict = {}
    for n_order in newton_model_dict.keys():
        newton_model = newton_model_dict[n_order]
        y_pred_total_list = []
        newton_loss_list = []
        for _ in range(max_num_step + 1):
            y_pred_total = torch.zeros(xs.shape[0], device=xs.device)  # [N, n]
            y_pred_total_list.append(y_pred_total)

        y_pred_list = newton_model(xs.double(), ys.double(), max_num_steps=max_num_step)
        for j in range(max_num_step + 1):
            y_pred_total_list[j] = y_pred_list[j].cpu()

        for j in range(max_num_step + 1):
            newton_loss = (ys[:, -1].cpu() - y_pred_total_list[j]).square()  # [N,]
            newton_loss_list.append(newton_loss.cpu())
        newton_loss_dict[n_order] = torch.stack(newton_loss_list, dim=1)  # [N, n_steps]
    return aggregate_metrics(newton_loss_dict, d)


def get_tf_loss(args, model, xs, ys, w_true=None):
    batch_size = 100  # for memory issue
    sample_size, n_points = xs.shape[0], xs.shape[1]
    assert sample_size % batch_size == 0
    model.eval()
    with torch.no_grad():
        if args.training.task_name in ["logistic_regression"]:
            y_pred_total = torch.zeros(sample_size, xs.shape[-1], device=xs.device)  # [N, d]
        else:
            y_pred_total = torch.zeros(sample_size, device=xs.device)  # [N, n]
        for batch_idx in range(sample_size // batch_size):
            xs_train = xs[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            ys_train = ys[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            y_pred = model(xs_train, ys_train, act_clip=args.training.activation_threshold)  # [B, n]
            y_pred_total[batch_idx * batch_size: (batch_idx + 1) * batch_size] = y_pred.detach()
    if args.training.task_name in ["logistic_regression"]:
        tf_loss = (w_true[:, :, 0] - y_pred_total).square().mean(1)
    else:
        tf_loss = (ys[:, -1] - y_pred_total).square()
    return tf_loss, y_pred_total


def init_device(args):
    cuda = args.gpu.cuda
    gpu = args.gpu.n_gpu
    if cuda:
        device = torch.device("cuda:{}".format(gpu))
    else:
        device = torch.device("cpu")
        torch.set_num_threads(4)
    return device


def rm_orig_mod(state_dict):
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict


def load_pretrained_model(args, model, state_path, device):
    if os.path.exists(state_path[:-3]):
        os.rename(state_path[:-3], state_path)
    if os.path.exists(state_path):
        state_dict = torch.load(state_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        flag = True
    else:
        print("train from scratch")
        flag = False
    return model, flag


def load_model(args, state_path, n_layer, device):
    from models import build_model
    args.model.n_layer = n_layer
    model = build_model(args.model)
    model.to(device)

    # If the model is trained already, then load the model and return
    print("load from: ", state_path + '.pt')
    model, flag_pretrained = load_pretrained_model(args, model, state_path + ".pt", device)
    assert flag_pretrained is True

    return model


def get_run_id(args):
    now = datetime.datetime.now().strftime('%m%d%H%M%S')
    run_id = "{}-{}-".format(now, args.wandb.name) + str(uuid.uuid4())[:4]
    return run_id
