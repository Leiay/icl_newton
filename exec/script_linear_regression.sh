config_file="configs/base.yaml"

for std in 0.0
do
  for n_cond in 1 5 10
  do
    python train.py --config $config_file \
        --wandb.name "lsa_cond={$n_cond}_std={$std}" \
        --training.cond $n_cond \
        --training.std $std \
        --gpu.n_gpu 0
  done
done &


config_file="configs/lsa_ln.yaml"
for std in 0.0
do
  for n_cond in 1 5 10
  do
    python train.py --config $config_file \
        --wandb.name "lsa_ln_cond={$n_cond}_std={$std}" \
        --training.cond $n_cond \
        --training.std $std \
        --gpu.n_gpu 1
  done
done &

