import torch
import torch.nn as nn


def build_model(conf):
    model = TransformerModel(
        n_dims=conf.n_dims,
        n_positions=conf.n_positions,
        n_embd=conf.n_embd,
        n_layer=conf.n_layer,
        n_head=conf.n_head,
        d_zeros=conf.d_zeros,
        output_type=conf.output_type,
        model_type=conf.model_type,
        rm_pos_embd=conf.rm_pos_embd,
    )

    return model


class BaseModel(nn.Module):
    def __init__(self, d_zeros):

        super(BaseModel, self).__init__()
        self.d_zeros = d_zeros

    def _combine(self, xs_b, ys_b):
        """
        :param xs_b: shape [B, n, d_in]
        :param ys_b: shape [B, n]
        :return: shape [B, n, d_zeros + d_in + 1]
        """
        B, n, d = xs_b.shape

        zs = torch.cat(
            (
                torch.zeros(B, n, self.d_zeros, device=xs_b.device),
                xs_b,
                ys_b.view(B, n, 1),
            ),
            axis=2,
        )

        zs[:, -1, -1] = 0.0  # set the last y_n to be 0.
        return zs

    def forward(self, xs, ys):
        raise NotImplementedError


class TransformerModel(BaseModel):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, d_zeros=0,
                 output_type='y', model_type='lsa', rm_pos_embd=False):

        super(TransformerModel, self).__init__(d_zeros)
        if model_type == 'lsa':
            from backbone_network.lsa_backbone import GPT2Model, GPT2Config
        elif model_type == 'lsa+ln':
            from backbone_network.tf_mod_backbone import GPT2Model, GPT2Config
        elif model_type == 'tf':
            from backbone_network.tf_backbone import GPT2Model, GPT2Config
        else:
            raise NotImplementedError
        configuration = GPT2Config()

        configuration.n_embd = n_embd
        self.combine_func = self._combine
        input_dim = n_dims + d_zeros + 1
        configuration.block_size = n_positions + 1
        configuration.n_layer = n_layer
        configuration.n_head = n_head
        configuration.dropout = 0.0
        configuration.bias = True
        self.configuration = configuration
        self.rm_pos_embd = rm_pos_embd

        self._read_in = nn.Linear(input_dim, n_embd)
        self._backbone = GPT2Model(self.configuration)
        self.output_type = output_type
        if output_type == 'y':
            self._read_out = nn.Linear(n_embd, 1)
        elif output_type == 'w':
            self._read_out = nn.Linear(n_embd, n_dims)
        else:
            raise NotImplementedError

    def forward(self, xs=None, ys=None, act_clip=-1, **kwargs):
        """
        :param xs: [B, n, d]
        :param ys: [B, n]
        :return:
        """
        zs = self.combine_func(xs, ys)  # [B, n, d_in], [B, n], -> [B, n, d_in + 1]

        inputs_embeds = self._read_in(zs)
        f_output = self._backbone(
            inputs_embeds=inputs_embeds,
            rm_pos_embd=self.rm_pos_embd,
            act_clip=act_clip
        )  # [B, n, d_in + 1]
        prediction = self._read_out(f_output)  # [B, 2n, d] -> [B, 2n, 1]

        if self.output_type == 'y':
            pred = prediction[:, -1, 0]  # [B,]
        elif self.output_type == 'w':
            pred = prediction[:, -1, :]  # [B, d]
        else:
            raise NotImplementedError

        return pred