import os
import datetime

from quinine import QuinineArgumentParser
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import random
import yaml
import numpy as np

from schema import schema
from models import build_model
from tasks import get_task_sampler
from main_utils import init_device, load_model, aggregate_metrics, get_tf_loss

import wandb

torch.backends.cudnn.benchmark = True


def logistic_func(w, X, Y, mu):
    """
    :param X: [B, N, d]
    :param Y: [B, N]
    :param w: [B, d, 1]
    """
    return ((- Y[:, :, None] * (X @ w)).exp() + 1).log().mean((1, 2)) + 0.5 * mu * w.square().sum((1, 2))


def main(args, device):
    # TORCH 2.0 ZONE ###############################
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    ################################################

    torch.manual_seed(4242)
    x_list = [0, 1, 6, 12, 20]
    x_list = x_list[::-1]

    # Result Directory ZONE ############################################################################################
    # # diff condition
    out_dirs = {
        "1": "0201230717-logistic_cond{1}-a471",
        "10": "0202144623-logistic_cond{10}-190a",
    }
    target_dir = "./results/logistic_regression/"
    mu = 0.01
    legend_keyword = 'cond'

    # # diff mu
    # out_dirs = {
    #     "0.1": "1015025653-new_d256_mu{0.1}-57ad",
    #     "0.01": "1015120516-new_d256_mu{0.01}-ca18",
    # }
    # target_dir = "../transformer_meta_learn/results/tf/"
    # cond_num = 10.0
    # legend_keyword = 'mu'

    # END Result Directory ZONE ########################################################################################

    model_dir = {}
    task_dir = {}
    for keys in out_dirs.keys():
        args.out_dir = target_dir + out_dirs[keys]
        model_list = []

        if legend_keyword == 'cond':
            cond_num = float(keys)
        elif legend_keyword == 'mu':
            mu = float(keys)

        task_sampler = get_task_sampler(
            task_name=args.training.task_name,
            batch_size=args.training.batch_size,
            n_points=args.training.points,  # curriculum.n_points,
            n_dims=args.model.n_dims,
            n_dims_truncated=args.model.n_dims,
            device=device,
            cond_num=cond_num,
            std=args.training.std,
            flag_train=False,
            logistic_mu=mu,
            logistic_pred_which=args.training.logistic_pred_which,
        )

        for n_layer in x_list:
            state_path = os.path.join(args.out_dir, f"layer_{n_layer}_cond_{cond_num}.0_std_{std}")
            model = load_model(args, state_path, n_layer=n_layer, device=device)
            model_list.append(model)

        model_dir[keys] = model_list
        task_dir[keys] = task_sampler

    # evaluate
    newton_err_dict = {}
    tf_err_dict, tf_to_newton_dict, newton_to_newton_dict = {}, {}, {}
    for keys in model_dir.keys():
        task_sampler = task_dir[keys]
        task = task_sampler()
        xs, ys = task.xs.float(), task.ys.float()
        ws = task.w_b.float()

        if legend_keyword == 'mu':
            mu = float(keys)

        # plot for Newton
        newton_err_list = []
        print("Caculate logistic reg with mu:", mu)
        ws_newton_list = []
        for step in range(max(x_list) + 1):
            print(step)
            task.newton(step)
            ws_newton = task.w_newton.float()
            ws_newton_list.append(ws_newton)
            reg_logistic = logistic_func(ws_newton, xs, ys, mu)
            newton_err_list.append(reg_logistic.cpu())
        newton_err_dict[keys] = torch.stack(newton_err_list, dim=1)
        # after this, ws_newton has already becomes the newton solution with max(x_list) steps.
        # it should have already converged

        # plot for TF
        model_list = model_dir[keys]
        tf_err_list, tf_to_newton_list = [], []
        newton_to_newton_list = []
        for idx, model in enumerate(model_list):
            tf_loss, ws_tf = get_tf_loss(args, model, xs, ys, ws)
            reg_logistic = logistic_func(ws_tf[:, :, None], xs, ys, mu)
            tf_err_list.append(reg_logistic.cpu())
            tf_to_newton_list.append((ws_tf - ws_newton[:, :, 0]).square().mean(1).cpu())

        for _ws_newton in ws_newton_list:
            newton_to_newton_list.append((_ws_newton[:, :, 0] - ws_newton[:, :, 0]).square().mean(1).cpu())

        tf_err_arr = torch.stack(tf_err_list, dim=1)
        tf_to_newton_arr = torch.stack(tf_to_newton_list, dim=1)
        newton_to_newton_arr = torch.stack(newton_to_newton_list, dim=1)
        tf_err_dict[keys] = tf_err_arr
        tf_to_newton_dict[keys] = tf_to_newton_arr
        newton_to_newton_dict[keys] = newton_to_newton_arr

    newton_err_dict_agg = aggregate_metrics(newton_err_dict, 1)
    tf_err_dict_agg = aggregate_metrics(tf_err_dict, 1)
    tf_to_newton_dict_agg = aggregate_metrics(tf_to_newton_dict, 1)
    newton_to_newton_dict_agg = aggregate_metrics(newton_to_newton_dict, 1)

    # start plotting ######################################################
    color_list = ['#073B4C', '#EF476F', '#118AB2', '#06D6A0', '#FFD166', '#4D4C7D', '#FFDAC0', '#F54952', ]

    # start plotting
    how_many_layers = 1  # TODO: checking how many layers.
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for idx, keys in enumerate(tf_err_dict.keys()):
        ax.semilogy(x_list, tf_err_dict_agg[keys]['mean'], linestyle='-', color=color_list[idx],
                    label=f'{legend_keyword}={keys}', linewidth=2, )
        low = tf_err_dict_agg[keys]["bootstrap_low"]
        high = tf_err_dict_agg[keys]["bootstrap_high"]
        ax.fill_between(x_list, low, high, alpha=0.3, color=color_list[idx], )

        ax.semilogy(np.arange(max(x_list) + 1) * how_many_layers, newton_err_dict_agg[keys]['mean'],
                    linestyle='--', color=color_list[idx], linewidth=2)  # , label='Newton,mu={}'.format(keys))
        low = newton_err_dict_agg[keys]["bootstrap_low"]
        high = newton_err_dict_agg[keys]["bootstrap_high"]
        ax.fill_between(np.arange(max(x_list) + 1) * how_many_layers, low, high, alpha=0.3, color=color_list[idx], )

    ax.set_xlim([0, 20])
    ticks = [0.05, 0.1, 0.2, 0.4, 0.8]

    ax.legend(loc='best', fontsize=13, ncol=1)
    ax.set_title("Logistic Regression Loss", fontsize=15)
    ax.grid(color='grey', linestyle='--')
    ax.set_xlabel('layers / steps', fontsize=15)
    ax.set_yticks(ticks, ticks)
    ax.tick_params(axis='both', labelsize=12)

    if not os.path.exists('results/Figs'):
        os.mkdir('results/Figs')
    if not os.path.exists('results/Figs/logistic_regression'):
        os.mkdir('results/Figs/logistic_regression')

    plt.savefig(
        f"results/Figs/logistic_regression/loss_how_many_layers_{how_many_layers}_{legend_keyword}.pdf",
        bbox_inches='tight', dpi=300)
    plt.close()

    # start plotting
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    for idx, keys in enumerate(tf_err_dict.keys()):
        ax.semilogy(x_list, tf_to_newton_dict_agg[keys]['mean'], linestyle='-', color=color_list[idx],
                    linewidth=5, label=f'{legend_keyword}={keys}')
        low = tf_to_newton_dict_agg[keys]["bootstrap_low"]
        high = tf_to_newton_dict_agg[keys]["bootstrap_high"]
        ax.fill_between(x_list, low, high, alpha=0.3, color=color_list[idx], )

    ax.grid(color='grey', linestyle='--')
    ax.set_title(r"Distance to $\mathbf{w}_{\mathrm{Newton}}$", fontsize=15)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("layers", fontsize=15)
    plt.savefig(f"results/Figs/logistic_regression/w_err_{legend_keyword}.pdf",
                bbox_inches='tight',
                dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    print(f"Running with: {args}")

    # device = init_device(args)
    device = torch.device("cpu")
    torch.set_num_threads(4)

    main(args, device)
