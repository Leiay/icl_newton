config_file="configs/logistic_regression.yaml"

mu=0.01
for n_cond in 1 10 50
do
  python train.py --config $config_file \
      --wandb.name "logistic_cond{$n_cond}" \
      --training.cond $n_cond \
      --training.logistic_mu $mu \
      --gpu.n_gpu 0
done

#n_cond=10
#for mu in 0.1 0.01 0.001
#do
#  python train.py --config $config_file \
#      --wandb.name "logistic_mu{$mu}" \
#      --training.cond $n_cond \
#      --training.logistic_mu $mu \
#      --gpu.n_gpu 0
#done
#
#
#n_cond=10
#mu=0.01
#for logistic_pred_which in "ground_truth" "newton"
#do
#  python train.py --config $config_file \
#      --wandb.name "logistic_pred_$logistic_pred_which" \
#      --training.cond $n_cond \
#      --training.logistic_mu $mu \
#      --training.logistic_pred_which $logistic_pred_which \
#      --gpu.n_gpu 0
#done
