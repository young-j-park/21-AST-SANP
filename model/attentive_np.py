
import torch.nn as nn

from model.neural_process import NP


class ANP(NP):
    def _build_nets(self):
        super()._build_nets()
        self.SA_r = nn.Sequential(
            nn.TransformerEncoderLayer(
                d_model=self.emb_dim, nhead=2, dim_feedforward=self.emb_dim
            ),
            nn.TransformerEncoderLayer(
                d_model=self.emb_dim, nhead=2, dim_feedforward=self.emb_dim
            )
        )
        self.CA_r = nn.MultiheadAttention(
            embed_dim=self.emb_dim, num_heads=2
        )

    def xy_to_r(self, x_emb, y_emb, x_target):
        r_i = self.encoder_r(x_emb + y_emb)
        r_i = self.SA_r(r_i.transpose(0, 1))
        r_attn, _ = self.CA_r(
            x_target.transpose(0, 1), x_emb.transpose(0, 1), r_i
        )
        return r_attn.transpose(0, 1)
