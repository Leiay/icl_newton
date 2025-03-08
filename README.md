# How Well Can Transformers Emulate In-Context Newton's Method?
Angeliki Giannou, Liu Yang, Tianhao Wang, Dimitris Papailiopoulos, Jason D. Lee

You can find the paper in [arxiv](https://arxiv.org/abs/2403.03183).

## Overview
Transformer-based models have demonstrated remarkable in-context learning capabilities, prompting extensive research into its underlying mechanisms. Recent studies have suggested that Transformers can implement first-order optimization algorithms for in-context learning and even second order ones for the case of linear regression. In this work, we study whether Transformers can perform higher order optimization methods, beyond the case of linear regression. We establish that linear attention Transformers with ReLU layers can approximate second order optimization algorithms for the task of logistic regression and achieve ϵ error with only a logarithmic to the error more layers. As a by-product we demonstrate the ability of even linear attention-only Transformers in implementing a single step of Newton's iteration for matrix inversion with merely two layers. These results suggest the ability of the Transformer architecture to implement complex algorithms, beyond gradient descent.

In the codebase, we include the implementation for the encoder Transformer, linear self-attention (LSA), 
and linear self-attention with LayerNorm, which learns to solve
(1) linear regression with different condition number, as well as (2) logistic regression.
The backbone transformer code is based on [NanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py), 
while the prompt generation code is based on 
[Garg et al.](https://github.com/dtsip/in-context-learning/tree/main)'s codebase.

```
@article{giannou2024well,
  title={How Well Can Transformers Emulate In-context Newton's Method?},
  author={Giannou, Angeliki and Yang, Liu and Wang, Tianhao and Papailiopoulos, Dimitris and Lee, Jason D},
  journal={arXiv preprint arXiv:2403.03183},
  year={2024}
}
```

## Environment Installation
Set up the environment by running:
```shell
conda env create -f env.yml
conda activate icl_newton
```

## Training Instructions

### Linear Regression
To train the linear regression models, execute:
```bash
bash exec/script_linear_regression.sh
```

Model configurations:
- `configs/base.yaml`: Self-attention only model.
- `configs/lsa_ln.yaml`: Self-attention model with LayerNorm.

Important hyperparameters in config files:
- `n_cond`: Condition number for the covariance matrix.
- `std`: Noise level of the labels.

### Logistic Regression
To train the logistic regression model, execute:
```bash
bash exec/script_logistic_regression.sh
```

Important hyperparameters:
- `mu`: Regularization parameter.
- `n_cond`: Condition number for the covariance matrix.

## Plotting Results

To plot the results, use the provided script:
```bash
bash exec/script_plot.sh
```

This script generates:
- Linear function plots and model error for linear regression tasks.
- MSE loss plots for logistic regression tasks.

Before plotting, ensure the output directories (`out_dirs`) in `script_plot.sh` are correctly set:
```python
# Result Directory ZONE ##################
out_dirs = {...}  # Specify your results directory here
# END Result Directory ZONE ##############
```


