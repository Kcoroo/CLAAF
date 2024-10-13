import torch
import torch.nn as nn
from transformers import RobertaModel, CLIPTextModel, ResNetModel, CLIPVisionModel


import torch
from torch import nn, einsum
from einops import rearrange, repeat


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNormForward(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormAttention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        return self.fn(q, k, v)


class PreNormAHL(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, h_t, h_v, h_hyper):
        h_t = self.norm1(h_t)
        h_v = self.norm3(h_v)
        h_hyper = self.norm4(h_hyper)

        return self.fn(h_t, h_v, h_hyper)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class HhyperLearningLayer(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_q_tv = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_tv = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_tv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=True),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, h_t, h_v, h_hyper):
        b, n, _, h = *h_t.shape, self.heads

        q = self.to_q(h_t)
        q_tv = self.to_q_tv(h_v)
        k_tv = self.to_k_tv(h_v)
        v_tv = self.to_v_tv(h_v)

        q, q_tv, k_tv, v_tv = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, q_tv, k_tv, v_tv))

        # tv self attention
        self_dots_tv = einsum('b h i d, b h j d -> b h i j', q_tv, k_tv) * self.scale
        self_attn_tv = self.attend(self_dots_tv)
        self_out_tv = einsum('b h i j, b h j d -> b h i d', self_attn_tv, v_tv)
        self_out_tv = rearrange(self_out_tv, 'b h n d -> b n (h d)')

        dots_tv = einsum('b h i d, b h j d -> b h i j', q, k_tv) * self.scale
        attn_tv = self.attend(dots_tv)
        out_tv = einsum('b h i j, b h j d -> b h i d', attn_tv, v_tv)
        out_tv = rearrange(out_tv, 'b h n d -> b n (h d)')

        h_hyper_shift = self.to_out(out_tv + self_out_tv)
        h_hyper += h_hyper_shift

        return h_hyper


class HhyperLearningEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAHL(dim, HhyperLearningLayer(dim, heads = heads, dim_head = dim_head, dropout = dropout))
            ]))

    def forward(self, h_t_list, h_v, h_hyper):
        for i, attn in enumerate(self.layers):
            h_hyper = attn[0](h_t_list[i], h_v, h_hyper)
        return h_hyper


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAttention(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, save_hidden=False):
        if save_hidden == True:
            hidden_list = []
            hidden_list.append(x)
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
                hidden_list.append(x)
            return hidden_list
        else:
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
            return x


class CrossTransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAttention(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, source_x, target_x):
        for attn, ff in self.layers:
            target_x_tmp = attn(target_x, source_x, source_x)
            target_x = target_x_tmp + target_x
            target_x = ff(target_x) + target_x
        return target_x


class Transformer(nn.Module):
    def __init__(self, *, num_frames, token_len, save_hidden, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.token_len = token_len
        self.save_hidden = save_hidden

        if token_len is not None:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + token_len, dim))
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
        else:
             self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
             self.extra_token = None

        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()


    def forward(self, x):
        b, n, _ = x.shape

        if self.token_len is not None:
            extra_token = repeat(self.extra_token, '1 n d -> b n d', b = b)
            x = torch.cat((extra_token, x), dim=1)
            x = x + self.pos_embedding[:, :n+self.token_len]
        else:
            x = x + self.pos_embedding[:, :n]

        x = self.dropout(x)
        x = self.encoder(x, self.save_hidden)

        return x


class CrossTransformer(nn.Module):
    def __init__(self, *, source_num_frames, tgt_num_frames, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.pos_embedding_s = nn.Parameter(torch.randn(1, source_num_frames + 1, dim))
        self.pos_embedding_t = nn.Parameter(torch.randn(1, tgt_num_frames + 1, dim))
        self.extra_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.CrossTransformerEncoder = CrossTransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

    def forward(self, source_x, target_x):
        b, n_s, _ = source_x.shape
        b, n_t, _ = target_x.shape

        extra_token = repeat(self.extra_token, '1 1 d -> b 1 d', b = b)

        source_x = torch.cat((extra_token, source_x), dim=1)
        source_x = source_x + self.pos_embedding_s[:, : n_s+1]

        target_x = torch.cat((extra_token, target_x), dim=1)
        target_x = target_x + self.pos_embedding_t[:, : n_t+1]

        source_x = self.dropout(source_x)
        target_x = self.dropout(target_x)

        x_s2t = self.CrossTransformerEncoder(source_x, target_x)

        return x_s2t



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.h_hyper = nn.Parameter(torch.ones(1, 8, 128))
        self.roberta = RobertaModel.from_pretrained('./models/roberta-base')
        # self.clip = CLIPTextModel.from_pretrained('./models/clip-vit-base')
        self.resnet = ResNetModel.from_pretrained('./models/resnet-50')
        # self.clip = CLIPVisionModel.from_pretrained('./models/clip-vit-base')
        # self.model = EMT(dim=32, depth=2, heads=4, num_modality=3)
        # self.gcn = GCNLayer(n_dim=32)
        # self.transformer = Transformer(d_model=768)
        self.text_fc = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU()
        )
        self.img_fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU()
        )

        self.proj_text = Transformer(num_frames=128, save_hidden=False, token_len=8, dim=128, depth=1, heads=8, mlp_dim=128)
        self.proj_v = Transformer(num_frames=49, save_hidden=False, token_len=8, dim=128, depth=1, heads=8, mlp_dim=128)

        self.text_encoder = Transformer(num_frames=8, save_hidden=True, token_len=None, dim=128, depth=2, heads=8, mlp_dim=128)
        self.h_hyper_layer = HhyperLearningEncoder(dim=128, depth=3, heads=8, dim_head=16, dropout = 0.)
        self.fusion_layer = CrossTransformer(source_num_frames=8, tgt_num_frames=8, dim=128, depth=2, heads=8, mlp_dim=128)


        self.fc1 = nn.Sequential(
            nn.Linear(768, 2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 2)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(2048, 2)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, image_inputs):
        h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b = input_ids.size(0))

        text_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # x_fea = self.clip(input_ids=input_ids, attention_mask=attention_mask)
        img_out = self.resnet(image_inputs)
        # img_fea = self.clip(image_inputs)
        # cls_out = x_fea.last_hidden_state[::,0]
        text_fea = text_out.last_hidden_state
        text_fea = self.text_fc(text_fea) # b, seq_len, 128
        test_cls_out = text_fea[::,0] # b, 128
        img_fea = img_out.last_hidden_state
        img_fea = img_fea.view(img_fea.shape[0], img_fea.shape[1], -1).transpose(1, 2).contiguous()
        img_fea = self.img_fc(img_fea) # b, img_size, 128
        img_cls_out = img_fea.mean(1) # b, 128
        # cls_out = x_fea.last_hidden_state[::,0]

        h_v = self.proj_v(img_fea)[:, :8]
        h_t = self.proj_text(text_fea)[:, :8]
        h_t_list = self.text_encoder(h_t)
        h_hyper = self.h_hyper_layer(h_t_list, h_v, h_hyper)
        feat = self.fusion_layer(h_hyper, h_t_list[-1])[:, 0]
        
        cls_out = feat
        
        
        return cls_out, self.fc4(cls_out)





