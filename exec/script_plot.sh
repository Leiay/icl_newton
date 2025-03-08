########################################################################################################################
### Plot linear function
### For lsa
#python plot_linear_func.py --config configs/base.yaml \
#    --gpu.n_gpu 1 \
#    --training.batch_size 1000

## For lsa w/ ln
#python plot_linear_func.py --config configs/lsa_ln.yaml \
#    --gpu.n_gpu 1 \
#    --training.batch_size 1000

########################################################################################################################
### Plot linear regression loss
# For lsa
# python plot_linear_regression_loss.py --config configs/base.yaml \
#    --gpu.n_gpu 1 \
#    --training.batch_size 1000

### For lsa w/ ln
#python plot_linear_regression_loss.py --config configs/lsa_ln.yaml \
#    --gpu.n_gpu 0 \
#    --training.batch_size 1000

########################################################################################################################
### Plot for logistic regression result
#mu=0.01
#python plot_logistic_regression.py --config configs/logistic_regression.yaml \
#    --gpu.n_gpu 0 \
#    --training.logistic_mu $mu \
#    --training.batch_size 2000
