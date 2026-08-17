"""Qwen3-VL-Embedding-2B 推理(pooling / 纯 prefill),单文件自包含。

本模块把 OpenSpec 拆成子包的 vision tower、多模态(MRoPE 位置 + 图像预处理)、
pooling head 全部折叠进一个文件,保持小而可跑。world_size>1 时通过既有 TP 线性层
+ TP-aware 的 VL 加载器(utils/loader_vl.py)做张量并行;vision tower / DeepstackProj /
EmbeddingHead 复制(残差流在每 rank 都是 full)。

关键调研结论(实现前已核对,见 openspec design R-1/R-3/R-4):
- R-1:checkpoint 无投影层;1_Pooling/config.json = pooling_mode:"lasttoken",
  embedding_dimension:2048。故 EmbeddingHead = last-token gather + L2(+可选 MRL)。
- R-3:flash prefill kernel 原本是因果的(attention.py:181 mask_causal),ViT 双向
  注意力不能用,故 Attention 加了 IS_CAUSAL 开关,ViT 用 is_causal=False。
- R-4:MRoPE 位置语义从 transformers get_rope_index 移植:文本 token 三轴 T=H=W
  随 current_pos 前进;图像 token 的 T/H/W 从 patch 网格派生,current_pos 推进
  max(H,W)//spatial_merge。频率布局交错 [T0,H0,W0,...,T19,H19,W19,T20..T23]。
- DeepStack 修正:视觉特征在 ViT 层 [5,11,17] 抽取,但注入到"文本 decoder 层 [0,1,2]"
  (不是 5,11,17),且只在图像 token 位置加(transformers _deepstack_process)。
"""
import itertools
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from myvllm.layers import (
    MRotaryEmbedding,          # 文本 decoder 的多模态 RoPE(3D T/H/W)
    Attention,                 # 复用项目 flash 内核;ViT 用 is_causal=False
    LayerNorm,                 # 项目里 LayerNorm 实为 RMSNorm
    MergedColumnParallelLinear,  # gate_up 合并 + TP 列分片
    QKVColumnParallelLinear,   # QKV 合并 + GQA 感知 TP 头切分
    RowParallelLinear,         # o_proj / down_proj 行分片 + all_reduce
    SiluAndMul,                # SwiGLU 激活
    VocabParallelEmbedding,    # embed_tokens 词表分片
    VisionRotaryEmbedding,     # ViT 的 RoPE(theta=10000,非交错)
)
from myvllm.utils import get_context

IMAGE_TOKEN_ID = 151655  # <|image_pad|>,用于在 input_ids 里定位图像 token

# ============================ vision tower ============================


class PatchEmbed3D(nn.Module):
    """3D patch embedding:Conv3d 在 (C, T, H, W) 上,步长 = kernel =
    (temporal_patch_size, patch_size, patch_size),每个时空 patch 产出一个
    hidden_size 维 token。忠实移植 transformers Qwen3VLVisionPatchEmbed。"""

    def __init__(self, patch_size=16, temporal_patch_size=2, in_channels=3, hidden_size=1024):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = hidden_size
        kernel = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, hidden_size, kernel_size=kernel, stride=kernel, bias=True)

    def forward(self, x):
        # x: (num_patches, C*T_patch*P*P) 已按 patch 摊平;reshape 回卷积输入形状
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(x.to(self.proj.weight.dtype)).view(-1, self.embed_dim)


class PatchMerger(nn.Module):
    """2×2 相邻 patch 合并:4*hidden -> out_hidden,忠实移植 transformers
    Qwen3VLVisionPatchMerger。use_postshuffle_norm 控制 norm 作用在合并后还是
    合并前(deepstack merger 用 True,主 merger 用 False)。"""

    def __init__(self, hidden_size, out_hidden_size, spatial_merge_size=2, use_postshuffle_norm=False):
        super().__init__()
        self.merged = hidden_size * (spatial_merge_size ** 2)  # 2×2=4 路拼接
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.merged if use_postshuffle_norm else hidden_size, eps=1e-6)
        self.fc1 = nn.Linear(self.merged, self.merged)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.merged, out_hidden_size)

    def forward(self, x):
        # 先按需 norm,再 reshape 成 (seq, 4*hidden),最后 MLP 投影到 out_hidden
        x = self.norm(x.view(-1, self.merged) if self.use_postshuffle_norm else x).view(-1, self.merged)
        return self.fc2(self.act(self.fc1(x)))


def _rotate_half(t):
    """GPT-NeoX 式 rotate_half:把最后一维对半切,返回 cat(-后半, 前半)。
    与 transformers 的 rotate_half 一致,供 ViT RoPE 使用。"""
    a, b = t.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class VisionAttention(nn.Module):
    """ViT 自注意力(双向)。复用项目的 flash prefill 内核,经 Attention
    (is_causal=False)走非因果分支;旋转用 VisionRotaryEmbedding(theta=10000)。
    QKV 在 ViT 里是 bias=True 的合并 Linear(不分片,ViT 整塔复制)。"""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size)
        # is_causal=False → flash 内核不施加因果 mask,ViT 全 patch 双向 attend
        self.attn = Attention(num_heads, self.head_dim, num_kv_heads=num_heads, is_causal=False)
        # ViT rotary:dim=head_dim//2,2D (h,w) 位置(对齐 transformers Qwen3VLVisionRotaryEmbedding)
        self.rotary = VisionRotaryEmbedding(self.head_dim // 2, theta=10000.0)

    def forward(self, x, position_ids, cu_seqlens):
        s = x.shape[0]
        # (s, 3*dim) -> (3, s, num_heads, head_dim) -> unbind 成 q,k,v
        qkv = self.qkv(x).reshape(s, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
        q, k, v = qkv.unbind(0)
        # ViT RoPE:cat(freqs,freqs) 后取 cos/sin,rotate-half 旋转
        emb = self.rotary(position_ids)
        cos, sin = emb.cos(), emb.sin()
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # (s,1,head_dim) 广播到多头
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        o = self.attn(q, k, v)  # (s, num_heads*head_dim)
        return self.proj(o)


class VisionBlock(nn.Module):
    """一个 ViT block:LN→attn→残差→LN→MLP→残差。MLP 用 gelu_pytorch_tanh。"""

    def __init__(self, hidden_size, intermediate_size, num_heads):
        super().__init__()
        self.n1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.n2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = VisionAttention(hidden_size, num_heads)
        self.mlp_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.mlp_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x, position_ids, cu_seqlens):
        x = x + self.attn(self.n1(x), position_ids, cu_seqlens)
        x = x + self.mlp_fc2(self.act(self.mlp_fc1(self.n2(x))))
        return x


class VisionTower(nn.Module):
    """视觉塔:patch embed → 插值学习到的 2D 位置嵌入 → N 个 ViT block →
    merger(2×2→out_hidden)。返回 merged hidden + deepstack 特征列表
    (每个 deepstack_visual_indexes 一个,经 post-shuffle merger 产出)。

    注意:整塔在 TP 下复制(每 rank 各跑完整 ViT,输出相同),避免 ViT 内部
    all_reduce;ViT 体量小(1024×24 层),复制成本可接受。"""

    def __init__(self, depth=24, hidden_size=1024, intermediate_size=4096, num_heads=16,
                 patch_size=16, temporal_patch_size=2, in_channels=3, out_hidden_size=2048,
                 spatial_merge_size=2, num_position_embeddings=2304,
                 deepstack_visual_indexes=None):
        super().__init__()
        self.sms = spatial_merge_size
        self.grid = int(num_position_embeddings ** 0.5)  # 48x48 学习位置网格
        self.dsi = deepstack_visual_indexes or []         # [5,11,17]
        self.patch_embed = PatchEmbed3D(patch_size, temporal_patch_size, in_channels, hidden_size)
        self.pos_embed = nn.Embedding(num_position_embeddings, hidden_size)
        self.blocks = nn.ModuleList([VisionBlock(hidden_size, intermediate_size, num_heads) for _ in range(depth)])
        self.merger = PatchMerger(hidden_size, out_hidden_size, spatial_merge_size, False)
        # 每个 deepstack 层一个 post-shuffle merger(把 ViT 中间特征 2×2 合并 + 投影到 out_hidden)
        self.dsm = nn.ModuleList([PatchMerger(hidden_size, out_hidden_size, spatial_merge_size, True) for _ in self.dsi])

    def _pos(self, h, w, dev, dt):
        """把 48×48 学习位置网格双线性插值到 (h, w),返回 (h*w, hidden)。"""
        n = self.grid
        p = self.pos_embed(torch.arange(self.pos_embed.num_embeddings, device=dev)).view(n, n, -1)
        p = p.permute(2, 0, 1).unsqueeze(0)  # (1,hidden,48,48)
        p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=True)
        return p.squeeze(0).permute(1, 2, 0).reshape(h * w, -1).to(dt)

    def _merge2x2(self, h, shapes):
        """把每张图的 (t,h,w,hidden) 按 2×2 折叠成 (t,h/2,w/2,4*hidden) 再 flatten。
        用于主 merger 与 deepstack merger 之前的特征重组。"""
        out, off = [], 0
        for (t, h_, w_) in shapes:
            n = t * h_ * w_
            x = h[off:off + n].view(t, h_, w_, -1).view(t, h_ // 2, 2, w_ // 2, 2, -1)
            out.append(x.permute(0, 1, 3, 2, 4, 5).reshape(t * (h_ // 2) * (w_ // 2), -1))
            off += n
        return torch.cat(out, dim=0)

    def forward(self, pixel_values, grid_thw):
        h = self.patch_embed(pixel_values)  # (total_patches, hidden)
        dev, dt = h.device, h.dtype
        # grid_thw 以 patch 为单位:(t, h, w),h、w 被 spatial_merge 整除
        shapes = [(int(t), int(hh), int(ww)) for (t, hh, ww) in grid_thw.tolist()]
        # 加插值后的学习 2D 位置嵌入(每张图全 patch 网格)
        pos = torch.cat([self._pos(hh, ww, dev, dt) for (t, hh, ww) in shapes], dim=0)
        h = h + pos
        # ViT rotary 用 2D (hpos, wpos) 位置(行主序,对齐 Conv3d 的 patch 顺序);
        # 每图 meshgrid(arange(h), arange(w)) 拼成 (h*w, 2),按 t 帧重复(v1 t=1)。
        pos2_list = []
        for (t, hh, ww) in shapes:
            hp, wp = torch.meshgrid(torch.arange(hh, device=dev), torch.arange(ww, device=dev), indexing="ij")
            pos2 = torch.stack([hp.flatten(), wp.flatten()], dim=-1)  # (hh*ww, 2)
            pos2_list.append(pos2.repeat(t, 1))                       # (t*hh*ww, 2)
        pids = torch.cat(pos2_list, dim=0)                            # (total_patches, 2)
        sizes = [t * hh * ww for (t, hh, ww) in shapes]
        # cu_seqlens 划分各图边界,供 varlen flash 注意力
        cu = torch.zeros(len(sizes) + 1, dtype=torch.int32, device=dev)
        for i, s in enumerate(sizes):
            cu[i + 1] = cu[i] + s
        # 跑 N 个 ViT block;命中 deepstack 索引时用 post-shuffle merger 抽特征
        feats = []
        for ln, blk in enumerate(self.blocks):
            h = blk(h, pids, cu)
            if ln in self.dsi:
                feats.append(self.dsm[self.dsi.index(ln)](self._merge2x2(h, shapes)))
        # 主 merger 产出最终 merged hidden(序列长度 = 总 patch / 4)
        return self.merger(self._merge2x2(h, shapes)), feats


# ============================ multimodal (MRoPE 位置 + 数据) ============================


def _vision_pos_ids(start, grid_thw, sms, dev):
    """移植 transformers Qwen3VLModel.get_vision_position_ids(图像分支,
    time_interval=1)。返回 (3, t*h_m*w_m) 的 T/H/W 位置。
    T = arange(t)*1;H = arange(h_m)+start;W = arange(w_m)+start;最后 T 再加 start。"""
    t, h, w = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
    gt, gh, gw = t, h // sms, w // sms  # 空间按 spatial_merge 缩
    pt = torch.arange(gt, device=dev)
    ph = torch.arange(gh, device=dev) + start
    pw = torch.arange(gw, device=dev) + start
    T, H, W = torch.meshgrid(pt, ph, pw, indexing="ij")
    v = torch.stack([T, H, W], dim=0).reshape(3, -1)
    v[0] += start  # T 偏移在 time_interval 乘之后加(此处 time_interval=1 无影响)
    return v


def _positions_single(input_ids, token_types, grids, sms, dev):
    """移植 transformers get_rope_index(单序列,无 attention_mask)。
    按连续同 token_type 分组:文本(0)三轴都等于 arange+offset;图像(1)/视频(2)
    用 _vision_pos_ids,current_pos 推进 max(H,W)//sms。返回 (3, seq)。"""
    gi = iter(grids)
    groups = []
    for key, grp in itertools.groupby(enumerate(token_types.tolist()), lambda x: x[1]):
        grp = list(grp)
        groups.append((key, grp[0][0], grp[-1][0] + 1))
    cur = 0
    parts = []
    for mt, s, e in groups:
        if mt == 0:  # 文本:T=H=W=arange+offset,三轴同步前进
            n = e - s
            parts.append(torch.arange(n, device=dev).view(1, -1).expand(3, -1) + cur)
            cur += n
        else:  # 图像/视频:从 patch 网格派生 T/H/W
            g = next(gi)
            parts.append(_vision_pos_ids(cur, g, sms, dev))
            cur += max(int(g[1]), int(g[2])) // sms  # 推进较大空间维
    return torch.cat(parts, dim=1)


def compute_mrope_positions(seqs, spatial_merge_size=2):
    """批打包的 MRoPE 位置。seqs 是 list of
    {"input_ids":1D, "token_types":1D, "image_grids":list[(t,h,w)]}。
    每条序列的 current_pos 从 0 起,拼成 (3, total_tokens)。"""
    dev = seqs[0]["input_ids"].device if seqs else torch.device("cpu")
    return torch.cat([_positions_single(s["input_ids"], s["token_types"], s["image_grids"], spatial_merge_size, dev) for s in seqs], dim=1)


@dataclass
class MultimodalData:
    """挂在 Sequence 上的每请求多模态载荷。v1 只图像:pixel_values(HF processor
    摊平的 patch)、grid_thw(每图 t,h,w patch 数)、token_types(mm_token_type_ids:
    text=0/image=1/video=2,给 get_rope_index 用)、image_token_spans(input_ids 里
    <|image_pad|> 的连续 (start,end),用于把视觉 embedding scatter 进对应位置)。"""

    pixel_values: torch.Tensor | None = None
    grid_thw: torch.Tensor | None = None
    token_types: torch.Tensor | None = None
    image_token_spans: list = field(default_factory=list)

    @property
    def has_image(self):
        return self.pixel_values is not None and self.pixel_values.numel() > 0


def build_multimodal_inputs(processor, instruction, item, image_token_id=IMAGE_TOKEN_ID):
    """用 HF AutoProcessor 把 {"text","image"} 请求变成 input_ids /
    mm_token_type_ids / pixel_values / image_grid_thw,并扫出 image_token_spans
    (连续 <|image_pad|> 的 (start,end))。lazy 导入 qwen_vl_utils,保证没装 VL 依赖时
    引擎其余部分仍可 import。"""
    content = []
    if item.get("image"):
        content.append({"type": "image", "image": item["image"]})
    if item.get("text"):
        content.append({"type": "text", "text": item["text"]})
    if not content:
        content.append({"type": "text", "text": ""})
    # 系统指令(instruction-aware,默认 "Represent the user's input.")+ 用户内容
    msgs = [{"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content}]
    # add_generation_prompt=True(per sentence_bert_config),取最后一个 token 做 pooling
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    images = None
    if item.get("image"):
        from qwen_vl_utils import process_vision_info
        imgs, _ = process_vision_info([{"image": item["image"]}])
        images = [imgs[0]]
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
    ids = inputs["input_ids"][0]
    tt = inputs.get("mm_token_type_ids")
    tt = tt[0] if tt is not None else None
    pv = inputs.get("pixel_values")
    gt = inputs.get("image_grid_thw")
    # 扫 input_ids 找连续 image_token_id 区段
    lst, i, n = [], 0, ids.tolist()
    while i < len(n):
        if n[i] == image_token_id:
            j = i
            while j < len(n) and n[j] == image_token_id:
                j += 1
            lst.append((i, j)); i = j
        else:
            i += 1
    return {"input_ids": ids, "token_types": tt, "pixel_values": pv, "grid_thw": gt, "image_token_spans": lst}


# ============================ pooling head ============================


def mrl_truncate(emb, dim):
    """Matryoshka 推理期截断:取前 dim 维并重新 L2 归一化。dim∈[64,2048]。"""
    if dim is None:
        return emb
    s = emb[..., :dim]
    return s / s.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class EmbeddingHead(nn.Module):
    """R-1 确认无投影层:仅 last-token gather + L2(+可选 MRL)。复制(每 rank
    相同,rank 0 返回)。last_token 用 cu_seqlens_q[1:]-1 取每序列最后 token
    (与 embedding_head.py:75-76 的 prefill-grep 同一招)。"""

    def __init__(self, pooling_mode="last_token", normalize=True, mrl_dim=None):
        super().__init__()
        self.mode = pooling_mode
        self.normalize = normalize
        self.mrl_dim = mrl_dim

    def forward(self, hidden, cu_seqlens_q):
        ns = cu_seqlens_q.shape[0] - 1
        if self.mode == "mean":
            out = [hidden[int(cu_seqlens_q[i]):int(cu_seqlens_q[i+1])].mean(0) for i in range(ns)]
            pooled = torch.stack(out, 0)
        else:  # last_token(默认)/ cls
            idx = (cu_seqlens_q[1:] - 1).to(hidden.device).long()
            pooled = hidden[idx] if self.mode != "cls" else hidden[cu_seqlens_q[:-1].to(hidden.device).long()]
        if self.normalize:
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return mrl_truncate(pooled, self.mrl_dim)


# ============================ text decoder (TP 分片) ============================


class Qwen3VLAttention(nn.Module):
    """文本 decoder 注意力:QKVColumnParallelLinear(GQA 感知 TP 头切片)+
    q_norm/k_norm(Qwen3 约定,attention_bias=False 时用)+ MRotaryEmbedding
    (3D 位置)+ Attention(因果)+ RowParallelLinear(all_reduce 出 full)。
    残差流在每 rank 都是 full hidden,故后续 DeepstackProj/EmbeddingHead 复制即可。"""

    def __init__(self, hidden_size, num_heads, head_dim, num_kv_heads, base, mrope_section, block_size):
        super().__init__()
        P = dist.get_world_size()
        self.num_heads = num_heads // P          # 每 rank 的 Q 头数
        self.num_kv_heads = num_kv_heads // P    # 每 rank 的 KV 头数(GQA)
        self.head_dim = head_dim
        self.q_size = head_dim * self.num_heads
        self.kv_size = head_dim * self.num_kv_heads
        # QKV 合并 + 按 GQA 头边界切(loaded_weight_id 'q'/'k'/'v' 在 loader 里用)
        self.qkv = QKVColumnParallelLinear(input_size=hidden_size, head_size=head_dim,
                                           num_heads=num_heads, num_kv_heads=num_kv_heads, bias=False)
        self.q_norm = LayerNorm(torch.ones(head_dim))
        self.k_norm = LayerNorm(torch.ones(head_dim))
        self.rotary = MRotaryEmbedding(base, head_dim, mrope_section or [24, 20, 20])
        # 文本 decoder 因果(LLM prefill);is_causal=True
        self.attn = Attention(self.num_heads, head_dim, num_kv_heads=self.num_kv_heads, block_size=block_size, is_causal=True)
        self.o = RowParallelLinear(input_size=head_dim * num_heads, output_size=hidden_size, bias=False)

    def forward(self, x, pos3d):
        q, k, v = self.qkv(x).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # varlen:(total_tokens, num_heads, head_dim)
        q = q.view(-1, self.num_heads, self.head_dim); k = k.view(-1, self.num_kv_heads, self.head_dim); v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rotary(pos3d, q, k)  # MRoPE:3D 位置 T/H/W
        return self.o(self.attn(q, k, v))  # all_reduce 后 full hidden


class Qwen3VLMLP(nn.Module):
    """SwiGLU MLP:gate_up 合并(TP 列分片)+ SiluAndMul + down 行分片(all_reduce)。"""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_up = MergedColumnParallelLinear(input_size=hidden_size, output_sizes=[intermediate_size] * 2, bias=True)
        self.act = SiluAndMul()
        self.down = RowParallelLinear(input_size=intermediate_size, output_size=hidden_size, bias=True)

    def forward(self, x):
        return self.down(self.act(self.gate_up(x)))


class Qwen3VLDecoderLayer(nn.Module):
    """input_layernorm → self_attn → 残差 → post_attention_layernorm → mlp → 残差。
    LayerNorm 支持 fused residual 变体(传 residual 时复用)。"""

    def __init__(self, hidden_size, num_heads, head_dim, num_kv_heads, intermediate_size, eps, base, mrope_section, block_size):
        super().__init__()
        g = torch.ones(hidden_size)
        self.in_ln = LayerNorm(g)
        self.post_ln = LayerNorm(g)
        self.attn = Qwen3VLAttention(hidden_size, num_heads, head_dim, num_kv_heads, base, mrope_section, block_size)
        self.mlp = Qwen3VLMLP(hidden_size, intermediate_size)

    def forward(self, x, residual, pos3d):
        if residual is not None:
            x, residual = self.in_ln(x, residual)  # fused residual+norm
        else:
            residual = x; x = self.in_ln(x)
        x = self.attn(x, pos3d)
        x, residual = self.post_ln(x, residual)
        x = self.mlp(x)
        return x, residual


# ============================ model ============================


class Qwen3VLForEmbedding(nn.Module):
    """Qwen3-VL-Embedding-2B:复制 vision tower + TP 分片的 28 层文本 decoder
    (DeepStack 视觉特征注入到"文本层 [0,1,2]",来自 ViT [5,11,17])+ 复制
    EmbeddingHead。无 LM head、无采样。"""

    packed_module_mapping = {"q_proj": ("q_proj", "q"), "k_proj": ("k_proj", "k"),
                             "v_proj": ("v_proj", "v"), "gate_up": ("gate_up_proj", "0"),
                             "gate_down": ("gate_down_proj", "1")}

    def __init__(self, vocab_size=151936, hidden_size=2048, num_heads=16, head_dim=128,
                 num_kv_heads=8, intermediate_size=6144, num_layers=28, rms_norm_epsilon=1e-6,
                 base=5_000_000, mrope_section=None, block_size=256, tie_word_embeddings=True,
                 vision_depth=24, vision_hidden_size=1024, vision_intermediate_size=4096,
                 vision_num_heads=16, patch_size=16, temporal_patch_size=2, in_channels=3,
                 out_hidden_size=2048, spatial_merge_size=2, num_position_embeddings=2304,
                 deepstack_visual_indexes=None, pooling_mode="last_token", normalize=True, mrl_dim=None):
        super().__init__()
        mrope_section = mrope_section or [24, 20, 20]
        self.dsi = deepstack_visual_indexes or [5, 11, 17]
        # vision tower(复制,见 VisionTower 注记)
        self.visual = VisionTower(vision_depth, vision_hidden_size, vision_intermediate_size, vision_num_heads,
                                  patch_size, temporal_patch_size, in_channels, out_hidden_size, spatial_merge_size,
                                  num_position_embeddings, self.dsi)
        # embed_tokens(词表 TP 分片,all_reduce 出 full)
        self.embed_tokens = VocabParallelEmbedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        # 28 层文本 decoder(TP 分片)
        self.layers = nn.ModuleList([Qwen3VLDecoderLayer(hidden_size, num_heads, head_dim, num_kv_heads,
                                  intermediate_size, rms_norm_epsilon, base, mrope_section, block_size) for _ in range(num_layers)])
        self.norm = LayerNorm(torch.ones(hidden_size))
        self.head = EmbeddingHead(pooling_mode, normalize, mrl_dim)

    def _forward_hidden(self, input_ids, pixel_values=None, grid_thw=None, image_token_spans=None):
        """跑到最终 RMSNorm 后的 hidden states(未过 head)。EmbeddingHead 与 lm_head
        共用此方法:Embedding 路径再 pooling,生成式路径再 lm_head。"""
        # input_ids:(total_tokens,) packed varlen;positions_3d 由 ModelRunner 经 context 传入
        pos3d = get_context().positions_3d
        x = self.embed_tokens(input_ids)
        img_idx = None
        if pixel_values is not None and image_token_spans:
            # 跑 vision tower(复制,每 rank 相同输出)→ (num_img_tokens, out_hidden) + deepstack 特征
            visual_emb, deepstack = self.visual(pixel_values, grid_thw)
            # 把视觉 embedding scatter 到 <|image_pad|> 位置(覆盖原 embedding,标准 Qwen3-VL 行为)
            ix = []
            for (s, e) in image_token_spans:
                ix.extend(range(s, e))
            img_idx = torch.tensor(ix, device=x.device, dtype=torch.long)
            x = x.clone(); x[img_idx] = visual_emb.to(x.dtype)
        else:
            deepstack = []
        # 文本 decoder;前 len(deepstack) 层(即 [0,1,2])注入对应 deepstack 特征(仅图像 token 位置)
        residual = None
        for li, layer in enumerate(self.layers):
            x, residual = layer(x, residual, pos3d)
            if li < len(deepstack) and img_idx is not None:
                # transformers _deepstack_process:hidden[visual_pos_masks] += visual_embeds
                x = x.clone(); x[img_idx] = x[img_idx] + deepstack[li].to(x.dtype)
        x, _ = self.norm(x, residual)
        return x

    def forward(self, input_ids, pixel_values=None, grid_thw=None, image_token_spans=None, cu_seqlens_q=None):
        # Embedding 路径:_forward_hidden → EmbeddingHead(last-token + L2 + MRL)
        x = self._forward_hidden(input_ids, pixel_values, grid_thw, image_token_spans)
        if cu_seqlens_q is None:
            cu_seqlens_q = torch.tensor([0, x.shape[0]], device=x.device, dtype=torch.long)
        # EmbeddingHead:last-token gather + L2(+MRL),复制,每 rank 相同
        return self.head(x, cu_seqlens_q)


# 生成式 VL 模型:复用 Embedding 的 vision/decoder/deepstack,把 EmbeddingHead 换成
# lm_head,forward 返回 hidden(供 run_model → compute_logits → 采样)。
# prefill 写 KV(图像 token 的 K/V 进 paged cache);decode 与纯文本一致(图像 token 的
# K/V 已在 cache,paged attention 读回)。MRoPE 位置对 decode 的新 token 取 (L,L,L) 近似
# (transformers 用 rope_deltas 调整,见 R-4;此处自洽可跑,严格数值对齐需 GPU 核对)。
class Qwen3VLForCausalLM(Qwen3VLForEmbedding):
    def __init__(self, vocab_size=151936, hidden_size=2048, num_heads=16, head_dim=128,
                 num_kv_heads=8, intermediate_size=6144, num_layers=28, rms_norm_epsilon=1e-6,
                 base=5_000_000, mrope_section=None, block_size=256, tie_word_embeddings=True,
                 vision_depth=24, vision_hidden_size=1024, vision_intermediate_size=4096,
                 vision_num_heads=16, patch_size=16, temporal_patch_size=2, in_channels=3,
                 out_hidden_size=2048, spatial_merge_size=2, num_position_embeddings=2304,
                 deepstack_visual_indexes=None):
        super().__init__(
            vocab_size=vocab_size, hidden_size=hidden_size, num_heads=num_heads, head_dim=head_dim,
            num_kv_heads=num_kv_heads, intermediate_size=intermediate_size, num_layers=num_layers,
            rms_norm_epsilon=rms_norm_epsilon, base=base, mrope_section=mrope_section,
            block_size=block_size, tie_word_embeddings=tie_word_embeddings,
            vision_depth=vision_depth, vision_hidden_size=vision_hidden_size,
            vision_intermediate_size=vision_intermediate_size, vision_num_heads=vision_num_heads,
            patch_size=patch_size, temporal_patch_size=temporal_patch_size, in_channels=in_channels,
            out_hidden_size=out_hidden_size, spatial_merge_size=spatial_merge_size,
            num_position_embeddings=num_position_embeddings,
            deepstack_visual_indexes=deepstack_visual_indexes,
            # EmbeddingHead 参数对生成式无意义,给默认即可(会被删除/替换)
            pooling_mode="last_token", normalize=True, mrl_dim=None)
        # 用 lm_head 替换 EmbeddingHead(词表 TP 分片 + gather 到 rank 0 采样,对标 qwen3.py)
        from myvllm.layers import ParallelLMHead
        self.lm_head = ParallelLMHead(num_embeddings=vocab_size, embedding_dim=hidden_size)
        if tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        del self.head  # 生成式不用 pooling head

    def forward(self, input_ids, pixel_values=None, grid_thw=None, image_token_spans=None, cu_seqlens_q=None):
        # 生成式:返回 hidden(未过 lm_head),由 ModelRunner.run_model 调 compute_logits
        return self._forward_hidden(input_ids, pixel_values, grid_thw, image_token_spans)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


if __name__ == "__main__":
    # 冒烟:纯文本前向(无 GPU,极小配置)。验证 import 与形状链路。
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from myvllm.utils import set_context
    set_context(is_prefill=True, runner_type="pooling", positions_3d=torch.zeros(3, 8, dtype=torch.long))
    m = Qwen3VLForEmbedding(num_layers=2, vision_depth=2)
    out = m(torch.randint(0, 151936, (8,)), cu_seqlens_q=torch.tensor([0, 8]))
    print(out.shape)
