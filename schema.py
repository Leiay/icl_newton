from quinine import (
    tstring,
    tinteger,
    tfloat,
    tboolean,
    stdict,
    tdict,
    tlist,
    default,
    required,
    allowed,
    nullable,
)
from funcy import merge


gpu_schema = {
    "cuda": merge(tboolean, default(True)),
    "n_gpu": merge(tinteger, required)  # which gpu device to use
}

model_schema = {
    "n_positions": merge(tinteger, required),  # maximum context length
    "n_dims": merge(tinteger, required),  # latent dimension
    "n_embd": merge(tinteger, required),
    "n_layer": merge(tinteger, required),
    "n_head": merge(tinteger, required),
    "d_zeros": merge(tinteger, required),
    "output_type": merge(tstring, default("y")),
    "model_type": merge(tstring, default("tf")),
    "rm_pos_embd": merge(tboolean, default(False)),
    "pretrained_path": merge(tstring, nullable, default(None)),
}

curriculum_base_schema = {
    "start": merge(tinteger, required),  # initial parameter
    "end": merge(tinteger, required),  # limit of final value
    "inc": merge(tinteger, required),  # how much to increment each time
    "interval": merge(tinteger, required),  # increment every how many steps
}

curriculum_schema = {
    "dims": stdict(curriculum_base_schema),
    "points": stdict(curriculum_base_schema),
}

training_schema = {
    "task_name": merge(tstring, required),
    "dims": merge(tinteger, required),
    "points": merge(tinteger, required),
    "std": merge(tfloat, default(0.)),
    "cond": merge(tfloat, default(0)),
    "logistic_mu": merge(tfloat, default(0.1)),
    "logistic_pred_which": merge(tstring, default("newton")),

    "batch_size": merge(tinteger, default(64)),
    "local_batch_size": merge(tinteger, default(64)),  # for memory efficiency

    "learning_rate": merge(tfloat, default(3e-4)),
    "min_lr": merge(tfloat, default(0)),
    "weight_decay": merge(tfloat, default(0.)),

    "grad_clip": merge(tfloat, default(-1)),  # if > 0, clip the gradient
    "activation_threshold": merge(tfloat, default(-1)),  # if > 0, clip the activation in each layer

    "train_steps": merge(tinteger, default(1000)),
    "warmup_steps": merge(tinteger, default(0)),
    "lr_decay_steps": merge(tinteger, default(10000)),
    "curriculum": stdict(curriculum_schema),
    "save_every_steps": merge(tinteger, default(1000)),  # how often to checkpoint
    "keep_every_steps": merge(tinteger, default(-1)),  # permanent checkpoints
    "resume_id": merge(tstring, nullable, default(None)),  # run uuid64
}


wandb_schema = {
    "project": merge(tstring, default("icl_newton")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
    "timestamp": merge(tstring, nullable)
}

schema = {
    "out_dir": merge(tstring, required),
    "gpu": stdict(gpu_schema),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "wandb": stdict(wandb_schema),
    "debug_mode": merge(tboolean, default(False)),
}
