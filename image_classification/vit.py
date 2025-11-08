# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
from torch import autograd
from torch.nn import functional as F
from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from torch.autograd import Variable

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        self.head = nn.Linear(self.head.in_features, 200)
        self.head_cb = nn.Linear(self.head.in_features, 200)
        self.contrast_head = nn.Sequential(nn.Linear(256, 256), )
        self.projection_head = nn.Sequential(nn.Linear(self.head.in_features, 256), )
        self.lamda = 40

    def forward_features(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome

    def forward(self, x, feat_flag=False,Aug=0):
        feat = self.forward_features(x)
        x = self.head(feat)
        if feat_flag: return x, feat
        if Aug == 4 :
            x = self.head(feat)
            out_cb = self.head_cb(feat)
            z = self.projection_head(feat)
            p = self.contrast_head(z)
            return x, out_cb, z, p
        return x

    def ewc_loss(self, cuda=False):
        # for name, param in self.named_parameters():
        #     print(f"Parameter name: {name}, Shape: {param.shape}")
        try:
            losses = []
            for n, p in self.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(self, '{}_mean'.format(n))
                fisher = getattr(self, '{}_fisher'.format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p - mean) ** 2).sum())
                print('loss:',(self.lamda / 2) * sum(losses))
            return (self.lamda / 2) * sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )

    def estimate_fisher(self, dataset, sample_size,scenario, batch_size=4):
        # sample loglikelihoods from the dataset.
        data_loader = scenario.build_dataloader(dataset, batch_size,0,True, False)
        loglikelihoods = []
        for x, y in data_loader:

            x = Variable(x).cuda() if self._is_on_cuda() else Variable(x)
            y = Variable(y).cuda() if self._is_on_cuda() else Variable(y)
            loglikelihoods.append(
                F.log_softmax(self(x), dim=1)[range(batch_size), y.data]
            )
            if len(loglikelihoods) >= sample_size // batch_size:
                break
        # estimate the fisher information of the parameters.
        loglikelihoods = torch.cat(loglikelihoods).unbind()
        # loglikelihood_grads = zip(*[autograd.grad(
        #     l, self.parameters(),
        #     retain_graph=(i < len(loglikelihoods))
        # ) for i, l in enumerate(loglikelihoods, 1)])
        for i, l in enumerate(loglikelihoods):
            print(f"loglikelihood {i}: requires_grad={l.requires_grad}, grad_fn={l.grad_fn}")
        loglikelihood_grads = []

        loglikelihood_grads = zip(*[autograd.grad(
                l,
                [p for p in self.parameters() if p.requires_grad],  # 只选择相关参数
                retain_graph=(i < len(loglikelihoods))
        )for i, l in enumerate(loglikelihoods, 1)])

        print('loglikelihood_grads：', loglikelihood_grads)
        for i, grad in enumerate(loglikelihood_grads):
            if grad is None:
                print(f"Element {i} is None")
        loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
        print('loglikelihood_grads：',loglikelihood_grads)
        fisher_diagonals = [(g ** 2).mean(0) for g in loglikelihood_grads]
        print('fisher_diagonals:',fisher_diagonals)
        param_names = [
            n.replace('.', '__') for n, p in self.named_parameters()
        ]
        print('param_names:',param_names)
        print('fish:',{n: f.detach() for n, f in zip(param_names, fisher_diagonals)})
        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

    def consolidate(self, fisher):
        print(f"Keys in fisher: {fisher.keys()}")

        for n, p in self.named_parameters():
            n = n.replace('.', '__')
            self.register_buffer('{}_mean'.format(n), p.data.clone())
            self.register_buffer('{}_fisher'
                                 .format(n), fisher[n].data.clone())
    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda



def vit_tiny_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_32(**kwargs):
    model = VisionTransformer(
        img_size=32,  # 输入图像的大小
        patch_size=4,  # 调整 patch 的大小，适应小尺寸图像
        in_chans=3,  # 输入图像的通道数
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_64(**kwargs):
    model = VisionTransformer(
        img_size=64,  # 输入图像的大小
        patch_size=8,  # 调整 patch 的大小，适应 ImageNet 样式
        in_chans=3,  # 输入图像的通道数
        embed_dim=192,  # 每个 patch 的嵌入维度
        depth=9,  # Transformer 的层数
        num_heads=12,  # 注意力头的数量
        mlp_ratio=4,  # MLP 的扩展比率
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_giant_patch14(**kwargs):
    model = VisionTransformer(
        img_size=64,
        patch_size=14,  # 14x14 的 Patch 大小
        embed_dim=1664,  # 更高的嵌入维度
        depth=48,  # Transformer 层数增加
        num_heads=16,  # 依然保持 16 头注意力
        mlp_ratio=4,  # MLP 扩展比率
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
