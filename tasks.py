import torch
import torch.nn.functional as F
import math


def get_task_sampler(
    task_name, batch_size, n_points, n_dims, n_dims_truncated, device, cond_num, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "noisy_linear_regression": NoisyLinearRegression,
        "logistic_regression": LinearClassification,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if task_name in ['linear_regression', 'noisy_linear_regression']:
            task_kwargs = {
                'value': kwargs.get('value', None),
                'flag_train': kwargs.get('flag_train', True),
            }
            if task_name == 'noisy_linear_regression':
                task_kwargs['std'] = kwargs.get('std', 0.1)
        else:
            task_kwargs = {
                'mu': kwargs.get('logistic_mu', 0.1),
                'logistic_pred_which': kwargs.get('logistic_pred_which', 'newton'),
                'flag_train': kwargs.get('flag_train', True),
            }
        return lambda **args: task_cls(
            task_name=task_name,
            batch_size=batch_size,
            n_points=n_points,
            n_dims=n_dims,
            n_dims_truncated=n_dims_truncated,
            device=device,
            cond_num=cond_num,
            **task_kwargs
        )
    else:
        print("Unknown task")
        raise NotImplementedError


class LinearRegression:

    def __init__(
            self, task_name, batch_size, n_points, n_dims, n_dims_truncated, device, cond_num,
            flag_train=True, value=None
    ):
        """
        `flag_train` and `value` is only for plotting the linear function.
        """
        super(LinearRegression, self).__init__()
        self.device = device
        self.n_dims = n_dims
        self.batch_size = batch_size
        self.n_dims_truncated = n_dims_truncated
        self.cond_num = cond_num

        if cond_num == 1:
            self.xs = torch.randn(batch_size, n_points, n_dims, device=device)  # [B, n, d]
        else:
            import torch.distributions as tdist
            self.xs = torch.empty(batch_size, n_points, n_dims, device=device)  # [B, n, d]

            while True:
                # Sometimes due to precision, we may encounter invalid Sigma matrices: not symmetric
                # In that case, we just try again
                A = torch.randn(n_dims, n_dims, device=device)
                U, S, V = torch.svd(A)
                max_eig = torch.rand(1, device=device) * 99 + 1
                min_eig = max_eig / cond_num
                other_eigs = torch.rand(n_dims - 2, device=device) * (max_eig - min_eig) + min_eig
                S[0] = max_eig
                S[n_dims - 1] = min_eig
                S[1:n_dims - 1] = torch.sort(other_eigs, descending=True)[0]
                Sigma = U @ torch.diag(S) @ U.t()
                try:
                    dist = tdist.multivariate_normal.MultivariateNormal(torch.zeros(n_dims, device=device), covariance_matrix=Sigma)
                    break
                except:
                    pass
            self.xs = dist.sample((batch_size, n_points,))  # [B, n, d]

        range_min = torch.quantile(self.xs, 0.005)
        range_max = torch.quantile(self.xs, 0.995)
        self.in_dist_x_range = (range_min, range_max)

        if not flag_train:
            # this is for plotting in linear regression
            if len(value) > 1:
                xs_test = value[1]
            else:
                xs_test = self.xs[0, -1, :]  # take the first test sample in the batch, shape [d]
            self.xs[:, -1, :] = xs_test[None, :]  # set all test samples to be the same
            self.xs[:, -1, 0] = value[0]  # set specific value for the first element

        w_b = torch.randn(batch_size, n_dims, 1, device=device)  # [B, d, 1]
        w_b[:, n_dims_truncated:] = 0
        self.w_b = w_b
        if not flag_train:
            # set the w_b to be the same for all samples
            if len(value) > 1:
                w_b_test = value[2]
            else:
                w_b_test = self.w_b[0, :, 0]  # take the first test sample in the batch, shape [d]
            self.w_b = w_b_test[None, :, None]  # set all test samples to be the same
        self.ys = (self.xs @ self.w_b).sum(-1)  # [B, n]


class NoisyLinearRegression(LinearRegression):
    def __init__(
            self, task_name, batch_size, n_points, n_dims, n_dims_truncated, device, cond_num, std, flag_train, value
    ):
        """
        :param std: the noise level
        """
        super(NoisyLinearRegression, self).__init__(
            task_name, batch_size, n_points, n_dims, n_dims_truncated, device, cond_num, flag_train, value)
        self.ys += torch.randn_like(self.ys) * std


class LinearClassification(LinearRegression):
    def __init__(
            self, task_name, batch_size, n_points, n_dims, n_dims_truncated, device, cond_num,
            flag_train=True, mu=0.1, logistic_pred_which='newton'
    ):
        """
        :param flag_train: this flag is used for either assigning the w_b label as w_newton (if flag_train=True),
                           or assigning the w_b label as ground truth (if flag_train=False). This flag will not affect
                           the linear regression class's flag_train argument, which by default is True.
        :param logistic_pred_which: use which as the label
        """
        super(LinearClassification, self).__init__(
            task_name, batch_size, n_points, n_dims, n_dims_truncated, device, cond_num)

        self.ys = self.ys.sign()  # ys are clean-labels, in {-1, 1}
        self.flag_train = flag_train
        self.mu = mu
        self.w_newton = None

        assert logistic_pred_which in ['newton', 'ground_truth']

        if flag_train:
            if logistic_pred_which == 'newton':
                self.newton()  # this will change the self.w_b to the newton's converged solution
                # while True:
                #     # sometimes the newton method produce error, so we need to try again
                #     try:
                #         self.newton()
                #         break
                #     except:
                #         pass
            else:
                pass

    def newton(self, step=-1):  # this is the batched version of newton

        def logistic_func(w, X, Y, mu):
            """
            :param X: [B, N, d]
            :param Y: [B, N, 1]
            :param w: [B, d]
            """
            return ((- Y * (X @ w[:, :, None])).exp() + 1).log().mean((1, 2)) + 0.5 * mu * w.square().sum(1)

        def one_step_newton_update(w_t, logistic_func, mu, X, Y):
            """
            :param w_t: shape [B, d,]
            :param mu: scalar
            :param logistic_func: a function that takes w_t and returns the loss with shape []
            """
            B = X.shape[0]
            jacobian = torch.autograd.functional.jacobian(lambda w: logistic_func(w, X, Y, mu), w_t)[
                torch.arange(B), torch.arange(B)]  # [B, d]

            # if we want to check again the jacobian calculation:
            # w_t.requires_grad=True
            # jac2 = torch.autograd.grad(logistic_func(w_t, X, Y, mu).sum(), w_t, create_graph=True)[0]
            # print((jacobian - jac2).abs().max())

            def get_sum_of_gradients(x):
                h = logistic_func(x, X, Y, mu).sum()
                return torch.autograd.grad(h, x, create_graph=True)[0].sum(0)

            hessian = torch.autograd.functional.jacobian(get_sum_of_gradients, w_t).swapaxes(0, 1)  # [B, d, d]
            lam_square = (jacobian[:, None, :] @ torch.linalg.inv(hessian) @ jacobian[:, :, None]).squeeze(1, 2)
            # [B, 1, d] @ [B, d, d] @ [B, d, 1] -> [B, 1, 1] -> [B,]
            lam = lam_square.sqrt()[:, None, None]  # [B, 1, 1]
            w_t_1 = w_t[:, :, None] - (2 * math.sqrt(mu) / (2 * math.sqrt(mu) + lam)) * torch.linalg.inv(
                hessian) @ jacobian[:, :, None]  # [d,]
            # [B, d, 1] - [B, 1, 1] * [B, d, d] @ [B, d, 1]
            return w_t_1[:, :, 0]  # [B, d]

        mu = self.mu
        # print("Use Mu:", mu)
        newton_learnt_w = torch.zeros_like(self.w_b)

        X = self.xs[:, :, :self.n_dims_truncated]  # [B, N, d]
        Y = self.ys[:, :, None]  # [B, N, 1]

        w_t = torch.randn_like(self.w_b[:, :self.n_dims_truncated, 0]) * 0.1  # initialization, shape [B, d]
        if step == -1:
            step = 30
        w_t_1 = w_t
        for i in range(step):
            w_t_1 = one_step_newton_update(w_t, logistic_func, mu, X, Y)
            mse_loss_t_1 = (self.w_b[:, :self.n_dims_truncated, 0] - w_t_1).square().mean(0).sum()
            mse_loss_t = (self.w_b[:, :self.n_dims_truncated, 0] - w_t).square().mean(0).sum()
            if (mse_loss_t_1 - mse_loss_t).abs() < 1e-6:
                # print("converged:", i)
                # print(logistic_func(w_t, X, Y, mu).mean(),
                #       logistic_func(self.w_b[:, :self.n_dims_truncated, 0], X, Y, mu).mean())
                break
            w_t = w_t_1
        newton_learnt_w[:, :self.n_dims_truncated, 0] = w_t_1
        if self.flag_train:
            self.w_b = newton_learnt_w
        else:
            self.w_newton = newton_learnt_w
