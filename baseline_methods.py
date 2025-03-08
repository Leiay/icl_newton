import torch
import math
import numpy as np
import os


# Gradient Descent and variants.
class GDModel:
    def __init__(
            self,
    ):
        super().__init__()

    def __call__(self, xs, ys, num_steps=1):
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        B = xs.shape[0]
        d = xs.shape[2]

        xs_train = xs[:, :-1]  # [B, n, d]
        ys_train = ys[:, :-1].view(B, -1, 1)  # [B, n, 1]
        xs_test = xs[:, -1:]  # [B, 1, d]
        w_0 = torch.zeros(B, d, 1).to(xs.device)
        test_pred = xs_test @ w_0

        cur_lr = 1.
        lr_search_list = []
        while cur_lr > 1e-8:
            lr_search_list.append(cur_lr)
            cur_lr = cur_lr * 0.8

        for i in range(num_steps):
            grad = xs_train.permute(0, 2, 1) @ (xs_train @ w_0 - ys_train)
            # [B, d, n] @ [B, n, 1] -> [B, d, 1]
            loss_list = []
            for idx, lr in enumerate(lr_search_list):
                w_1 = w_0 - lr * grad
                cur_loss = torch.mean((xs_train @ w_1 - ys_train) ** 2)
                loss_list.append(cur_loss)
            min_idx = loss_list.index(min(loss_list))
            lr = lr_search_list[min_idx]
            w_1 = w_0 - lr * grad  # [B, d, 1]
            test_pred = xs_test @ w_1
            w_0 = w_1
        return torch.squeeze(test_pred)


class NewtonOrderModel:
    def __init__(
            self,
            sigma=0.,
            n_order=10,
    ):
        super().__init__()
        self.sigma = sigma
        self.n_order = n_order

    def __call__(self, xs, ys, max_num_steps=1):
        B = xs.shape[0]
        d = xs.shape[2]

        xs = xs.cpu().double().numpy()
        ys = ys.cpu().double().numpy()

        xs_train = xs[:, :-1]  # [B, n, d]
        # A = np.transpose(xs_train, (0, 2, 1)) @ xs_train  # [B, d, d]
        A = np.matmul(np.transpose(xs_train, (0, 2, 1)), xs_train)  # [B, d, d]
        if self.sigma > 0:
            A = A + self.sigma * self.sigma * np.eye(d)[None, :, :]
        ys_train = ys[:, :-1].reshape(B, -1, 1)  # [B, n, 1]
        ys_test = ys[:, -1:].reshape(B, -1, 1)
        xs_test = xs[:, -1:]  # [B, 1, d]

        B = A @ np.transpose(A, (0, 2, 1))
        eig_val = np.linalg.eigvalsh(B)
        eig_max = np.max(np.real(eig_val), axis=1)  # [B,]
        eig_min = np.min(np.real(eig_val), axis=1)  # [B,]
        alpha = 1 / (eig_max + eig_min)[:, None, None]
        # alpha = 2 / np.abs(B).sum(-1).max(-1)[:, None, None]  # [B]
        z = np.transpose(A, (0, 2, 1)) * alpha  # init z, shape [B, d, d]
        w = z @ np.transpose(xs_train, (0, 2, 1)) @ ys_train  # [B, d, d] @ [B, d, n] @ [B, n, 1] -> [B, d, 1]
        test_pred = xs_test @ w  # [B, 1, 1]
        test_pred_list = [torch.squeeze(torch.from_numpy(test_pred), (1, 2))]

        for i in range(max_num_steps):
            # instead of doing: X_{k+1} = X_k(2I - AX_k)
            right_part = np.zeros_like(z)  # [B, d, d]
            for m in range(self.n_order):
                tmp = np.eye(d)[None, :, :]
                for j in range(m):
                    tmp = tmp @ (A @ z)
                right_part += ((-1) ** m) * math.comb(self.n_order, m + 1) * tmp
            z_new = z @ right_part
            w = z_new @ np.transpose(xs_train, (0, 2, 1)) @ ys_train  # [B, d, d] @ [B, d, n] @ [B, n, 1] -> [B, d, 1]
            test_pred = xs_test @ w  # [B, 1, 1]
            test_pred_list.append(torch.squeeze(torch.from_numpy(test_pred), (1, 2)))
            z = z_new

        return test_pred_list

