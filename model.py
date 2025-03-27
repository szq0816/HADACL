import copy
import torch
from HASF import Block as hasf
from utils import *
from graph_adjacency import *


class Discriminator(nn.Module):
    def __init__(self,
                 encoder_dim,
                 ):
        super(Discriminator, self).__init__()
        self.l = nn.Sequential(
            nn.Linear(encoder_dim[-1], 64),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.l(x)

class HADACL(torch.nn.Module):
    def __init__(self, batch_size, n_views, layer_dims, temperature, n_classes, drop_rate=0.5):
        super(HADACL, self).__init__()
        self.n_views = n_views
        self.n_classes = n_classes

        self.online_encoder = nn.ModuleList([FCN(layer_dims[i], drop_out=drop_rate) for i in range(n_views)])
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.discriminator = nn.ModuleList([Discriminator(layer_dims[i]) for i in range(n_views)])

        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.cross_view_decoder = nn.ModuleList([MLP(layer_dims[i][-1], layer_dims[i][-1]) for i in range(n_views)])
        self.cl = ContrastiveLoss(temperature)
        # self.sd_cl = ContrastiveLossWithSoftDistribution(temperature)
        self.feature_dim = [layer_dims[i][-1] for i in range(n_views)]
        self.atten2 = hasf(3, 128)
        self.cluster = ClusterProject(layer_dims[0][-1], self.n_classes)


    def forward(self, data, momentum, mask, warm_up, args):
        self._update_target_branch(momentum)
        z = [self.online_encoder[i](data[i][mask[:, i]]) for i in range(self.n_views)]
        z_g = [self.cross_view_decoder[i](z[i][mask[:, i]]) for i in range(self.n_views)]
        z_t = [self.target_encoder[i](data[i][mask[:, i]]) for i in range(self.n_views)]
        z_c = self.atten2(torch.stack([z_t[0], z_t[1]]))




        l_cl = (self.cl(z_c, z[0]) + self.cl(z_c, z[1])) * args.alpha + (self.cl(z[0], z_t[0]) + self.cl(z[1], z_t[1]))

        fake_z = [self.discriminator[i](z_g[i]) for i in range(self.n_views)]
        real_z = [self.discriminator[i](z_t[i]) for i in range(self.n_views)]
        d_loss = (torch.mean(nn.ReLU(inplace=True)(1.0 - real_z[0])) + torch.mean(
            nn.ReLU(inplace=True)(1.0 + fake_z[0]))) + \
                 (torch.mean(nn.ReLU(inplace=True)(1.0 - real_z[1])) + torch.mean(
                     nn.ReLU(inplace=True)(1.0 + fake_z[1])))
        l_cd = (self.cl(z_g[0], z_t[1]) + self.cl(z_g[1], z_t[0])) + d_loss * args.beta
        # l_sd = self.cl(sd_z0, sd_z1)
        loss = l_cl + l_cd
        # loss = l_inter + l_intra + loss_kl
        # print(loss)
        # print(l_cl)
        # print(l_cd)
        # print(d_loss)

        return loss


    @torch.no_grad()
    def _update_target_branch(self, momentum):
        for i in range(self.n_views):
            for param_o, param_t in zip(self.online_encoder[i].parameters(), self.target_encoder[i].parameters()):
                param_t.data = param_t.data * momentum + param_o.data * (1 - momentum)

    @torch.no_grad()
    def extract_feature(self, data, mask):
        N = data[0].shape[0]
        z = [torch.zeros(N, self.feature_dim[i]).cuda() for i in range(self.n_views)]
        for i in range(self.n_views):
            z[i][mask[:, i]] = self.target_encoder[i](data[i][mask[:, i]])
            # print(data[0][mask[:, 0]].shape)
        for i in range(self.n_views):
            z[i][~mask[:, i]] = self.cross_view_decoder[1 - i](z[1 - i][~mask[:, i]])
            # print(z[1 - 0][~mask[:, 0]].shape)

        z = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        # print(z[0].shape)
        z = [L2norm(z[i]) for i in range(self.n_views)]
        # z = self.atten2(torch.stack([z[0], z[1]]))
        # print(z.shape)

        return z


import torch.nn as nn
import torch.nn.functional as F
L2norm = nn.functional.normalize

class FCN(nn.Module):
    def __init__(self, dim_layer=None, norm_layer=None, act_layer=None, drop_out=0.0, norm_last_layer=True):
        super(FCN, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm1d
        layers = []
        for i in range(1, len(dim_layer) - 1):
            layers.append(nn.Linear(dim_layer[i - 1], dim_layer[i], bias=False))
            layers.append(norm_layer(dim_layer[i]))
            layers.append(act_layer())
            if drop_out != 0.0 and i != len(dim_layer) - 2:
                layers.append(nn.Dropout(drop_out))

        if norm_last_layer:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=False))
            layers.append(nn.BatchNorm1d(dim_layer[-1], affine=False))
        else:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=True))

        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        return self.ffn(x)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out=None, hidden_ratio=4.0, act_layer=None):
        super(MLP, self).__init__()
        dim_out = dim_out or dim_in
        dim_hidden = int(dim_in * hidden_ratio)
        act_layer = act_layer or nn.ReLU
        self.mlp = nn.Sequential(nn.Linear(dim_in, dim_hidden),
                                 act_layer(),
                                 nn.Linear(dim_hidden, dim_out))

    def forward(self, x):
        x = self.mlp(x)
        return x

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x_q, x_k, mask_pos=None):
        x_q = L2norm(x_q)
        x_k = L2norm(x_k)
        N = x_q.shape[0]
        if mask_pos is None:
            mask_pos = torch.eye(N).cuda()
        similarity = torch.div(torch.matmul(x_q, x_k.T), self.temperature)
        similarity = -torch.log(torch.softmax(similarity, dim=1))
        nll_loss = similarity * mask_pos / mask_pos.sum(dim=1, keepdim=True)
        loss = nll_loss.mean()
        return loss
