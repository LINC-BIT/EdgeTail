# vit_adapter_proto.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

# timm imports (assume timm is available in your environment)
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.vision_transformer import VisionTransformer, _create_vision_transformer, Block, Attention, LayerScale, checkpoint_filter_fn
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg, named_apply, adapt_input_conv, checkpoint_seq

# ----------------------------
# Basic Adapter (bottleneck)
# ----------------------------
class BasicAdapter(nn.Module):
    """
    Simple adapter bottleneck: down-project -> GELU -> up-project + residual.
    dim_down can be tuned; default uses dim//4 (min 8).
    """
    def __init__(self, dim, dim_down=None):
        super().__init__()
        if dim_down is None:
            dim_down = max(8, dim // 4)
        self.down = nn.Linear(dim, dim_down)
        self.act = nn.GELU()
        self.up = nn.Linear(dim_down, dim)
        # small init for residual branch
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        # x: (B, N, C)
        out = self.down(x)
        out = self.act(out)
        out = self.up(out)
        return x + out

# ----------------------------
# Attention used by classifier
# ----------------------------
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        # q: (B, Lq, d), k: (B, Lk, d), v: (B, Lv, d)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, dim=2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module (simple) '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        _, len_k, _ = k.size()
        _, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # (n*b) x lq x dk
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output

# ----------------------------
# Block with Adapter
# ----------------------------
class BlockWithAdapter(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.adapter1 = BasicAdapter(dim)

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.adapter2 = BasicAdapter(dim)

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.adapter1(self.attn(self.norm1(x)))))
        x = x + self.drop_path2(self.ls2(self.adapter2(self.mlp(self.norm2(x)))))
        return x

# ----------------------------
# ViTAdapter (from your code, slightly cleaned)
# ----------------------------
class ViTAdapter(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
            class_token=True, no_embed_class=False, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            weight_init='', embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=BlockWithAdapter, *args, **kwargs):

        # Mirror VisionTransformer init, but with block_fn defaulting to BlockWithAdapter
        super().__init__()  # keep base initialization minimal
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            try:
                self.init_weights(weight_init)
            except Exception:
                # fallback if base init not fully configured
                pass

    def forward(self, x):
        x_f = self.forward_features(x)
        x = self.forward_head(x_f)
        # return both features (cls token) and head logits
        return x_f[:, 0, :], x

# ----------------------------
# ProtoClassifier (based on your MYNET)

# ----------------------------
class DefaultArgs:
            def __init__(self):
                self.temperature = 10.0

class ProtoClassifier(nn.Module):
    """
    Prototype-based classifier using a small self-attention over (prototypes + query).
    Expects embeddings as inputs (no image->patch processing here).
    """

    def __init__(self, embed_dim, args=None):
        super().__init__()
        self.args = args



        self.args = DefaultArgs()
        # fallback temperature
        if not hasattr(self.args, 'temperature'):
            # default temperature scalar
            setattr(self.args, 'temperature', 10.0)
        self.slf_attn = MultiHeadAttention(1, embed_dim, embed_dim, embed_dim, dropout=0.5)

    def _forward(self, support, query):
        """
        support: (B, ways, shots, dim)  -> prototypes calculated as mean over shots
        query: (B, ways, n_query_per_way, dim) OR (B, n_query_total, dim)
               Accepts either (B, ways, nq, dim) OR flattened queries (B, nq_total, dim)
        """
        emb_dim = support.size(-1)
        proto = support.mean(dim=2)  # (B, ways, dim) assuming support shape (B, ways, shots, dim)
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]

        # normalize/reshape query
        if query.dim() == 4:
            # (B, ways, nquery, dim) -> flatten to (B, ways*nquery, dim)
            num_query = query.shape[1] * query.shape[2]
            query_flat = query.view(query.shape[0], -1, emb_dim)
        else:
            # assume already (B, n_q_total, dim)
            num_query = query.shape[1]
            query_flat = query

        # Expand proto to align with queries
        proto_exp = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
        proto_exp = proto_exp.view(num_batch * num_query, num_proto, emb_dim)

        # reshape queries: (B, n_q, dim) -> (B*n_q, 1, dim)
        query_exp = query_flat.view(num_batch * num_query, emb_dim).unsqueeze(1)

        # combined sequence: (N*K) x (N_proto + 1) x d
        combined = torch.cat([proto_exp, query_exp], dim=1)
        combined = self.slf_attn(combined, combined, combined)  # apply attention
        # split back
        proto_after, query_after = combined.split(num_proto, dim=1)  # proto_after: (BNq, num_proto, dim); query_after: (BNq, 1, dim)

        query_after = query_after.squeeze(1)  # (BNq, dim)
        # compute cosine similarity between query and each proto
        # proto_after: (BNq, num_proto, dim) -> expand query to match
        query_expand = query_after.unsqueeze(1).expand(-1, num_proto, -1)
        logits = F.cosine_similarity(query_expand, proto_after, dim=-1)  # (BNq, num_proto)
        logits = logits * self.args.temperature
        # reshape logits back to (B, n_q, num_proto)
        logits = logits.view(num_batch, num_query, num_proto)
        return logits

# ----------------------------
# ViT wrapper that integrates Adapter + ProtoClassifier
# ----------------------------
class ViTWithAdapterAndProto(ViTAdapter):
    """
    Combines ViTAdapter (which contains adapters inside Transformer blocks) with a ProtoClassifier head.
    Usage:
      model = ViTWithAdapterAndProto(cfg_args, img_size=224, patch_size=16, embed_dim=..., depth=..., num_heads=..., ...)
      model.mode = 'encoder'    # only return features
      or call model((support_imgs, query_imgs)) for few-shot inference:
        - support_imgs: tensor shaped (B * ways * shots, 3, H, W) or pre-extracted embeddings
        - query_imgs:  tensor shaped (B * ways * nquery, 3, H, W)
      The wrapper will extract embeddings then apply proto classification.
    NOTE: For convenience this expects you pass already-batched images; you can adapt to your dataset pipeline.
    """

    def __init__(self, args=None, *vit_args, **vit_kwargs):
            super().__init__(*vit_args, **vit_kwargs)

            # 使用 None 而不是 lambda 函数
            self.args = args

            embed_dim = getattr(self, 'embed_dim', getattr(self, 'num_features', None))
            if embed_dim is None:
                raise ValueError("Cannot infer embed_dim from ViT; ensure ViTAdapter constructed correctly.")

            self.proto = ProtoClassifier(embed_dim, args=self.args)

            # 设置默认模式，如果 args 为 None 则使用 None
            self.mode = getattr(self.args, 'mode', None) if self.args is not None else None

    def extract_embeddings_from_images(self, imgs):
        """
        imgs: (B_all, C, H, W)
        returns: (B_all, dim) - the cls token representation (first token)
        """
        # forward_features returns (B, N+1, C) or (B, N, C) depending on implementation; we used forward_features earlier.
        x_f = self.forward_features(imgs)  # (B_all, seq_len, dim)
        cls = x_f[:, 0, :]  # take cls token embedding
        return cls

    def forward(self, x):
        """
        Overloaded forward:
          - if mode == 'encoder' or self.mode == 'encoder': accept images -> return embeddings
          - else expect input = (support_imgs, query_imgs) where each are image tensors.
            We'll extract embeddings and reshape them to (B, ways, shots, dim) and (B, ways, nquery, dim)
            **Important**: This wrapper *assumes* you pass support & query already grouped by (B, ways, shots, C,H,W)
        """
        emb = self.extract_embeddings_from_images(x)  # (B, dim)

        # 传入分类头
        logits = self.head(emb)
        return logits

# ----------------------------
# Convenience factory functions (example)
# ----------------------------
def vit_tiny_patch16_224_with_mynet(args=None, pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, block_fn=BlockWithAdapter, **kwargs)
    model = ViTWithAdapterAndProto(args=args, **model_kwargs)
    return model

def vit_base_patch16_224_with_mynet(args=None, pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, block_fn=BlockWithAdapter, **kwargs)
    model = ViTWithAdapterAndProto(args=args, **model_kwargs)
    return model

def vit_huge_patch14_224_with_mynet(args=None, pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=224,
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        block_fn=BlockWithAdapter,
        **kwargs
    )
    model = ViTWithAdapterAndProto(args=args, **model_kwargs)
    return model


def vit_tiny_patch16_64_with_mynet(args=None, pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=64,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        block_fn=BlockWithAdapter,
        **kwargs
    )
    model = ViTWithAdapterAndProto(args=args, **model_kwargs)
    return model


def vit_base_patch16_64_with_mynet(args=None, pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=64,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        block_fn=BlockWithAdapter,
        **kwargs
    )
    model = ViTWithAdapterAndProto(args=args, **model_kwargs)
    return model


def vit_large_patch16_64_with_mynet(args=None, pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=64,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        block_fn=BlockWithAdapter,
        **kwargs
    )
    model = ViTWithAdapterAndProto(args=args, **model_kwargs)
    return model


def vit_huge_patch14_64_with_mynet(args=None, pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=64,
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        block_fn=BlockWithAdapter,
        **kwargs
    )
    model = ViTWithAdapterAndProto(args=args, **model_kwargs)
    return model