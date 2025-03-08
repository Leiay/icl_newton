import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from quinine import QuinineArgumentParser

from main_utils import init_device, aggregate_metrics, load_model, get_tf_loss
from models import build_model
from schema import schema
from tasks import get_task_sampler


color_list = ['#073B4C', '#EF476F', '#FFD166', '#4D4C7D', '#118AB2', '#06D6A0', '#F54952',]


def main(args,device):
    # TORCH 2.0 ZONE ###############################
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch.backends.cudnn.benchmark = True
    ################################################

    torch.manual_seed(4242)
    x_list = [1, 2, 3, 4, 5, 6]

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
        target_dir = "./results/lsa_linear/"
    else:
        # Normalized Attn ################################
        out_dirs = {
            1: '0131073853-lsa_ln_cond={1}_std={0.0}-f432',
            5: '0131114304-lsa_ln_cond={5}_std={0.0}-1423',
            10: '0131160614-lsa_ln_cond={10}_std={0.0}-c3d6',
        }
        target_dir = "./results/lsa_ln_linear/"
    # END Result Directory ZONE ########################################################################################

    value = 100  # so it will plot from -value to value
    inter_points = 100
    test_list = [-value + i * 2 * value / (inter_points - 1) for i in range(inter_points)]

    for n_cond_trained in out_dirs.keys():
        args.out_dir = target_dir + out_dirs[n_cond_trained]

        model_list = []

        for n_layer in x_list:
            state_path = os.path.join(
                args.out_dir, f"layer_{n_layer}_cond_{n_cond_trained}.0_std_{trained_std}")
            model = load_model(args, state_path, n_layer=n_layer, device=device)
            model_list.append(model)

        # for each model let's test it
        from baseline_methods import NewtonOrderModel
        newton_model_dict = {
            2: NewtonOrderModel(n_order=2, sigma=trained_std),
            3: NewtonOrderModel(n_order=3, sigma=trained_std),
        }
        num_newton_method = len(newton_model_dict.keys())
        fig, axs = plt.subplots(num_newton_method + 1,len(x_list), figsize=(14, 6))

        tf_pred_dict = {}
        tf_pred_dict_agg_total, newton_pred_dict_agg_total = {}, {}
        for idx_value, val in enumerate(test_list):
            print("The value is now:", val)
            if idx_value == 0:
                input_value = [val]
            else:
                input_value = [val, xs_test, w_b[0, :, 0]]

            task_sampler = get_task_sampler(
                task_name=args.training.task_name,
                batch_size=args.training.batch_size,
                n_points=args.training.points,  # curriculum.n_points,
                n_dims=args.model.n_dims,
                n_dims_truncated=args.model.n_dims,
                device=device,
                cond_num=n_cond_trained,
                std=trained_std,
                value=input_value,
                flag_train=False,
            )

            task = task_sampler()
            xs, ys = task.xs.float(), task.ys.float()
            w_b = task.w_b.float()
            xs_test = xs[0, -1, :]  # [d,]
            max_num_step = max(x_list)

            # Plot for Newton
            newton_loss_dict = {}
            for n_order in newton_model_dict.keys():
                newton_model = newton_model_dict[n_order]
                y_pred_total_list = []
                newton_loss_list = []

                y_pred_list = newton_model(xs.double(), ys.double(), max_num_steps=max_num_step)

                for j in range(max_num_step + 1):
                    y_pred_total_list.append(y_pred_list[j])  # [B, d]

                for j in range(max_num_step + 1):
                    newton_loss = y_pred_total_list[j]  # [:, 0]  # [B,]
                    newton_loss_list.append(newton_loss.cpu())
                newton_loss_dict[n_order] = torch.stack(newton_loss_list, dim=1)  # [B, n_steps]
            newton_loss_dict_agg = aggregate_metrics(newton_loss_dict, 1)

            if idx_value == 0:
                for idx in range(max_num_step + 1):
                    newton_pred_dict_agg_total[idx] = {}
                    for n_order in newton_model_dict.keys():
                        newton_pred_dict_agg_total[idx][n_order] = {}
                        newton_pred_dict_agg_total[idx][n_order]['mean'] = [newton_loss_dict_agg[n_order]['mean'][idx]]
                        newton_pred_dict_agg_total[idx][n_order]['std'] = [newton_loss_dict_agg[n_order]['std'][idx]]
                        newton_pred_dict_agg_total[idx][n_order]['bootstrap_low'] = [
                            newton_loss_dict_agg[n_order]['bootstrap_low'][idx]]
                        newton_pred_dict_agg_total[idx][n_order]['bootstrap_high'] = [
                            newton_loss_dict_agg[n_order]['bootstrap_high'][idx]]
            else:
                for idx in range(max_num_step + 1):
                    for n_order in newton_model_dict.keys():
                        newton_pred_dict_agg_total[idx][n_order]['mean'].extend([newton_loss_dict_agg[n_order]['mean'][idx]])
                        newton_pred_dict_agg_total[idx][n_order]['std'].extend([newton_loss_dict_agg[n_order]['std'][idx]])
                        newton_pred_dict_agg_total[idx][n_order]['bootstrap_low'].extend(
                            [newton_loss_dict_agg[n_order]['bootstrap_low'][idx]])
                        newton_pred_dict_agg_total[idx][n_order]['bootstrap_high'].extend(
                            [newton_loss_dict_agg[n_order]['bootstrap_high'][idx]])

            # plot for TF
            tf_pred_list = []
            for idx, model in enumerate(model_list):
                num_step = x_list[idx]
                _, tf_pred = get_tf_loss(args, model, xs, ys, w_b)

                tf_pred_list.append(tf_pred.cpu())
            tf_pred_arr = torch.stack(tf_pred_list, dim=1)
            tf_pred_dict['tf'] = tf_pred_arr

            tf_pred_dict_agg = aggregate_metrics(tf_pred_dict, 1)

            if idx_value == 0:
                for idx in range(len(x_list)):
                    tf_pred_dict_agg_total[idx] = {}
                    tf_pred_dict_agg_total[idx]['mean'] = [tf_pred_dict_agg['tf']['mean'][idx]]
                    tf_pred_dict_agg_total[idx]['std'] = [tf_pred_dict_agg['tf']['std'][idx]]
                    tf_pred_dict_agg_total[idx]['bootstrap_low'] = [tf_pred_dict_agg['tf']['bootstrap_low'][idx]]
                    tf_pred_dict_agg_total[idx]['bootstrap_high'] = [tf_pred_dict_agg['tf']['bootstrap_high'][idx]]
            else:
                for idx in range(len(x_list)):
                    tf_pred_dict_agg_total[idx]['mean'].extend([tf_pred_dict_agg['tf']['mean'][idx]])
                    tf_pred_dict_agg_total[idx]['std'].extend([tf_pred_dict_agg['tf']['std'][idx]])
                    tf_pred_dict_agg_total[idx]['bootstrap_low'].extend(
                        [tf_pred_dict_agg['tf']['bootstrap_low'][idx]])
                    tf_pred_dict_agg_total[idx]['bootstrap_high'].extend(
                        [tf_pred_dict_agg['tf']['bootstrap_high'][idx]])

        for idx, idx_x in enumerate(x_list):
            axs[0][idx].plot(test_list, tf_pred_dict_agg_total[idx]['mean'], linestyle='-', color=color_list[idx],
                        label=idx_x)
            low = tf_pred_dict_agg_total[idx]["bootstrap_low"]
            high = tf_pred_dict_agg_total[idx]["bootstrap_high"]
            axs[0][idx].fill_between(test_list, low, high, alpha=0.3, color=color_list[idx])
        for idx, idx_x in enumerate(x_list):
            for idx_order, n_order in enumerate(newton_pred_dict_agg_total[idx].keys()):
                axs[idx_order+1][idx].plot(test_list, newton_pred_dict_agg_total[idx][n_order]['mean'], linestyle='-', color=color_list[idx],
                            label=idx_x)
                low = newton_pred_dict_agg_total[idx][n_order]["bootstrap_low"]
                high = newton_pred_dict_agg_total[idx][n_order]["bootstrap_high"]
                axs[idx_order+1][idx].fill_between(test_list, low, high, alpha=0.3, color=color_list[idx])

                axs[idx_order+1][0].set_ylabel('Ord. {}'.format(n_order), fontsize=12)

        test_arr = torch.tensor(test_list)
        ground_true = test_arr * w_b[0, 0, 0].cpu() + (xs_test[1:] * w_b[0, 1:, 0]).cpu().sum()

        for i in range(len(x_list)):
            for j in range(num_newton_method + 1):
                axs[j][i].plot(test_list, ground_true, linestyle='--', color=color_list[0]) #, label='ground_truth')

        ylim = axs[0][0].get_ylim()
        for i in range(len(x_list)):
            for j in range(num_newton_method + 1):
                axs[j][i].set_ylim(ylim)
                axs[j][i].axvline(x=task.in_dist_x_range[0].cpu().numpy(), color='k', linestyle='--')
                axs[j][i].axvline(x=task.in_dist_x_range[1].cpu().numpy(), color='k', linestyle='--')

        for i in range(len(x_list)):
            axs[0][i].legend(loc='upper left')

        if args.model.model_type == 'lsa+ln':
            axs[0][0].set_ylabel('LSA w/ LN', fontsize=12)
        else:
            axs[0][0].set_ylabel('LSA', fontsize=12)

        if not os.path.exists('results/Figs'):
            os.mkdir('results/Figs')
        if not os.path.exists('results/Figs/linear_func'):
            os.mkdir('results/Figs/linear_func')
        plt.savefig(f"results/Figs/linear_func/range_{value}_model_{args.model.model_type}_cond={n_cond_trained}_std={trained_std}.pdf",
                    bbox_inches='tight')


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    print(f"Running with: {args}")

    device = init_device(args)

    if args.debug_mode:
        args.out_dir = "./results/debug"

    main(args, device)
