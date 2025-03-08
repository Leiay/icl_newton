import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from quinine import QuinineArgumentParser

from main_utils import init_device, load_model, aggregate_metrics
from schema import schema
from tasks import get_task_sampler


def main(args, device):
    # TORCH 2.0 ZONE ###############################
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch.backends.cudnn.benchmark = True
    ################################################

    # torch.manual_seed(4242)
    x_list = [1, 2, 3, 4, 5, 6,]
    x_list = x_list[::-1]

    from baseline_methods import NewtonOrderModel
    from main_utils import get_newton_loss, get_tf_loss

    # Result Directory ZONE ############################################################################################
    # the dictionary is with key = condition number, value = result_dir
    trained_std = 0.0  # this could be changed based on results
    if args.model.model_type == 'lsa':
        # LSA ############################################
        out_dirs = {
            1: "0131073853-lsa_cond={1}_std={0.0}-233b",
            5: "0131114506-lsa_cond={5}_std={0.0}-9641",
            10: "0131160439-lsa_cond={10}_std={0.0}-98ec"
        }
        label = "LSA"
        target_dir = "./results/lsa_linear/"
    else:
        # Normalized Attn ################################
        out_dirs = {
            1: '0131073853-lsa_ln_cond={1}_std={0.0}-f432',
            5: '0131114304-lsa_ln_cond={5}_std={0.0}-1423',
            10: '0131160614-lsa_ln_cond={10}_std={0.0}-c3d6',
        }
        label = "LSA w/ LN"
        target_dir = "./results/lsa_ln_linear/"
    # END Result Directory ZONE ########################################################################################

    model_dir = {}
    task_dir = {}
    for n_cond in out_dirs.keys():
        args.out_dir = target_dir + out_dirs[n_cond]
        model_list = []
        task_sampler = get_task_sampler(
            task_name=args.training.task_name,
            batch_size=args.training.batch_size,
            n_points=args.training.points,
            n_dims=args.model.n_dims,
            n_dims_truncated=args.model.n_dims,
            device=device,
            cond_num=n_cond,
            std=trained_std,
            flag_train=True,
        )

        for n_layer in x_list:
            state_path = os.path.join(
                args.out_dir, f"layer_{n_layer}_cond_{n_cond}.0_std_{trained_std}")

            model = load_model(args, state_path, n_layer=n_layer, device=device)
            model_list.append(model)

        model_dir[n_cond] = model_list
        task_dir[n_cond] = task_sampler

    # evaluate
    newton_model_dict = {
        2: NewtonOrderModel(n_order=2, sigma=trained_std),
        3: NewtonOrderModel(n_order=3, sigma=trained_std),
        4: NewtonOrderModel(n_order=4, sigma=trained_std),
        # 6: NewtonOrderModel(n_order=6, sigma=trained_std),
        # 8: NewtonOrderModel(n_order=8, sigma=trained_std),
        # 10: NewtonOrderModel(n_order=10, sigma=trained_std),
    }
    color_list = ['#073B4C', '#EF476F', '#06D6A0', '#118AB2', '#FFD166', '#4D4C7D', '#FFDAC0', '#F54952']

    for n_cond_trained in model_dir.keys():
        model_list = model_dir[n_cond_trained]
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

        task_sampler = task_dir[n_cond_trained]
        task = task_sampler()
        xs, ys = task.xs.float(), task.ys.float()
        w_b = task.w_b.float()

        # plot for Newton
        max_num_step = max(x_list)
        newton_err_dict = get_newton_loss(max_num_step, newton_model_dict, xs, ys)

        # plot for TF
        tf_err_list = []
        for idx, model in enumerate(model_list):
            tf_loss, _ = get_tf_loss(args, model, xs, ys, w_b)

            tf_err_list.append(tf_loss.cpu())
        tf_err_arr = torch.stack(tf_err_list, dim=1)
        tf_err_dict = {"tf": tf_err_arr}

        tf_err_dict_agg = aggregate_metrics(tf_err_dict, xs.shape[-1])

        # start plotting
        ax.semilogy(x_list, tf_err_dict_agg['tf']['mean'], linewidth=3, linestyle='-', color=color_list[0], label=label)
        low = tf_err_dict_agg['tf']["bootstrap_low"]
        high = tf_err_dict_agg['tf']["bootstrap_high"]
        ax.fill_between(x_list, low, high, alpha=0.3, color=color_list[0])
        for idx, n_order in enumerate(newton_err_dict.keys()):
            newton_err_agg = newton_err_dict[n_order]
            ax.semilogy(np.arange(len(newton_err_agg['mean'])), newton_err_agg['mean'], marker='*', linewidth=3,
                        linestyle='-', color=color_list[1 + idx], label='Newton(order={})'.format(n_order))
            low = newton_err_agg["bootstrap_low"]
            high = newton_err_agg["bootstrap_high"]
            ax.fill_between(np.arange(len(newton_err_agg['mean'])), low, high, alpha=0.3,
                            color=color_list[1 + idx])

        ax.legend(loc='best', fontsize=10.5, ncol=1)
        # ax.set_title("Linear Regression Error", fontsize=15)
        ax.set_title("Num Cond={}, noise={}".format(n_cond_trained, trained_std))
        ax.grid(color='grey', linestyle='--')
        ax.set_xlabel('layers / steps', fontsize=15)
        ax.tick_params(axis='both', labelsize=12)
        plt.xlim([min(x_list), max(x_list)])

        if not os.path.exists('results/Figs'):
            os.mkdir('results/Figs')
        if not os.path.exists('results/Figs/linear_regression_loss'):
            os.mkdir('results/Figs/linear_regression_loss')

        plt.savefig(
            f"results/Figs/linear_regression_loss/loss_{args.model.model_type}_trained={n_cond_trained}_std={trained_std}.pdf",
            bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    print(f"Running with: {args}")

    device = init_device(args)

    main(args, device)
