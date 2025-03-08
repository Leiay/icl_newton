import os
import datetime

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import random
import yaml

from schema import schema
from models import build_model
from tasks import get_task_sampler
from curriculum import Curriculum
from main_utils import init_device, get_run_id, CosineAnnealingWarmup

import wandb

torch.backends.cudnn.benchmark = True


def train_step(args, model, xs, ys, optimizer, w_true=None):
    batch_size = args.training.local_batch_size  # for memory issue
    sample_size, n_points = xs.shape[0], xs.shape[1]
    assert sample_size % batch_size == 0

    n_cumulative_step = sample_size // batch_size

    for batch_idx in range(n_cumulative_step):
        xs_train = xs[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        ys_train = ys[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        if args.training.task_name in ["logistic_regression"]:
            w_train = w_true[batch_idx * batch_size: (batch_idx + 1) * batch_size, :, 0]
            w_pred = model(xs_train, ys_train, act_clip=args.training.activation_threshold)  # [B, d]
            loss = (w_train - w_pred).square().mean()
        else:
            y_pred = model(xs_train, ys_train, act_clip=args.training.activation_threshold)  # [B, n]
            loss = (ys_train[:, -1] - y_pred).square().mean()  # take loss on the last sample

        (loss / n_cumulative_step).backward()

    if args.training.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.training.grad_clip)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.detach()


def train_tf_model(args, state_path, n_layer, device):

    args.model.n_layer = n_layer
    model = build_model(args.model)

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.training.learning_rate, weight_decay=args.training.weight_decay)

    lr_scheduler = CosineAnnealingWarmup(
        optimizer,
        warmup_steps=args.training.warmup_steps,
        learning_rate=args.training.learning_rate,
        min_lr=args.training.min_lr,
        lr_decay_steps=args.training.lr_decay_steps,
        verbose=False
    )

    curriculum = Curriculum(args.training.curriculum)

    wandb.init(
        dir=args.out_dir,
        project=args.wandb.project,
        config=args.__dict__,
        name=args.wandb.name + "_n_cond_" + str(args.training.cond) + \
             "_noise_std_" + str(args.training.std) + "_n_layer_" + str(n_layer),
        mode="disabled" if args.debug_mode else "online",
    )

    pbar = tqdm(range(0, args.training.train_steps))

    for i in pbar:
        task_sampler = get_task_sampler(
            task_name=args.training.task_name,
            batch_size=args.training.batch_size,
            n_points=curriculum.n_points,
            n_dims=args.model.n_dims,
            n_dims_truncated=curriculum.n_dims_truncated,
            device=device,
            cond_num=args.training.cond,
            std=args.training.std,
            flag_train=True,
            logistic_mu=args.training.logistic_mu,
            logistic_pred_which=args.training.logistic_pred_which,
        )

        task = task_sampler()
        xs, ys = task.xs.float(), task.ys.float()
        ws = task.w_b.float()

        loss = train_step(args, model, xs, ys, optimizer, ws)
        if i % args.wandb.log_every_steps == 0:
            wandb.log({
                "overall_loss": loss,
                "n_points": curriculum.n_points,
                "n_dims_truncated": curriculum.n_dims_truncated,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=i)

        # EVALUATION ======================================
        if lr_scheduler is not None:
            lr_scheduler.step()
        curriculum.update()

        if i % args.training.keep_every_steps == 0:
            torch.save(model.state_dict(), state_path + f"_step_{i}.pt", )

    wandb.finish()
    return model, task_sampler


def main(args, device):
    # TORCH 2.0 ZONE ###############################
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    ################################################

    torch.manual_seed(14)
    ## specify number of layers
    if args.training.task_name == "logistic_regression":
        x_list = [0, 1, 6, 12, 15, 20]  # layers for logistic regression
    else:
        x_list = [0, 1, 2, 3, 4, 5, 6]

    for n_layer in x_list:
        state_path = os.path.join(args.out_dir, f"layer_{n_layer}_cond_{args.training.cond}_std_{args.training.std}")
        model, _ = train_tf_model(args, state_path, n_layer=n_layer, device=device)
        torch.save(model.state_dict(), state_path + '.pt')


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    print(f"Running with: {args}")

    device = init_device(args)

    if args.debug_mode:
        args.out_dir = "./results/debug"

    run_id = args.training.resume_id
    if run_id is None:
        run_id = get_run_id(args)

    out_dir = os.path.join(args.out_dir, run_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    args.out_dir = out_dir
    # add a timestamp here, if resumed, this will be the resumed time
    args.wandb['timestamp'] = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
        yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args, device)

