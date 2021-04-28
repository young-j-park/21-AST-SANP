
import math

import torch.nn as nn
import torch.utils.data

from model.basic_modules import LinEmbedding, MLP
from utility.pytorch_utils import empty
from config import DEVICE


class NP(nn.Module):
    def __init__(
            self, x_dim=2, y_dim=1, out_dim=1, emb_dim=128,
            dist='Gaussian', stochastic=False, fill=False,
    ):
        super(NP, self).__init__()
        if stochastic:
            raise NotImplementedError

        self.x_dim = x_dim  # position
        self.y_dim = y_dim  # value
        self.out_dim = out_dim
        self.emb_dim = emb_dim
        self.stochastic = stochastic
        self.dist = dist
        self.fill = fill
        self._build_nets()
        self.to(DEVICE)

    def _1024_mlp(self):
        return MLP(self.emb_dim, [1024]*2, self.emb_dim)

    def _build_nets(self):
        self.encoder_x = LinEmbedding(self.x_dim, self.emb_dim)
        self.encoder_y = LinEmbedding(self.y_dim, self.emb_dim)

        self.encoder_r = MLP(self.emb_dim, [1024]*2, self.emb_dim)
        if self.stochastic:
            self.encoder_s = MLP(self.emb_dim, [1024]*2, self.emb_dim)
            self.encoder_z = MLP(self.emb_dim, [1024]*2, self.emb_dim*2)

        if self.dist == 'BCE':
            self.decoder = nn.Sequential(
                MLP(self.emb_dim*2, [1024]*2, self.y_dim),
                nn.Sigmoid()
            )
        elif self.dist == 'Gaussian':
            self.decoder = MLP(self.emb_dim*2, [1024]*2, self.out_dim*2)
        t = torch.randn(self.emb_dim) / math.sqrt(self.emb_dim)
        self.mask_emb = nn.Parameter(t.to(DEVICE))

    def xy_to_r(self, x_emb, y_emb, x_target):
        r_i = self.encoder_r(x_emb + y_emb)
        r = torch.mean(r_i, dim=1, keepdim=True).expand(-1, x_target.size(1), -1)
        return r

    def encode(self, y, mask, non_mask):
        y_emb = empty(mask.size(0), mask.size(1), self.emb_dim)
        y_emb[mask] = self.encoder_y(y[mask])
        y_emb[non_mask] = self.mask_emb
        return y_emb

    def decode(self, r_target, x_emb):
        z_x = torch.cat([r_target, x_emb], dim=-1)
        y_hat = self.decoder(z_x)
        return y_hat

    def forward(self, x_context, y_context, mask, non_mask, x_target=None):
        """
        :param x_context: (N, Tc, Dx)
        :param y_context: (N, Tc, Dy)
        :param mask: (N, Tc)
        :param non_mask: (N, Tc)
        :param x_target:  (N, Tt, Dx)
        :return: y_target:  (N, Tt, Dy)
        """
        # emb x
        x_emb = self.encoder_x(x_context)
        xt_emb = self.encoder_x(x_target) \
            if x_target is not None else x_emb

        # emb y
        y_emb = self.encode(y_context, mask, non_mask)

        # emb r
        r_target = self.xy_to_r(x_emb, y_emb, xt_emb)

        # decode y
        y_hat = self.decode(r_target, xt_emb)
        return y_hat
