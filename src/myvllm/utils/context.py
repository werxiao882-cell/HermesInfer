from dataclasses import dataclass 
import torch 


@dataclass
class Context:
    """单例 attention 元数据容器。每步由 ModelRunner 经 set_context() 写入、
    Attention/ParallelLMHead/decoder 经 get_context() 读,避免把元数据逐层传参。"""
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None     # varlen 各序列 Q 边界(累积长度)
    cu_seqlens_k: torch.Tensor | None = None      # varlen 各序列 K 边界
    max_seqlen_q: int = 0                         # 批内最长 Q(决定 flash grid)
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None      # KV 写入槽位(None→Attention 跳过 store)
    context_lens: torch.Tensor | None = None      # decode 每序列已生成长度
    block_tables: torch.Tensor | None = None      # decode 每序列的物理 block 映射
    # ---- VL-only(pooling / 多模态),文本生成路径不用 ----
    positions_3d: torch.Tensor | None = None      # (3, total_tokens) T/H/W,供 MRoPE
    image_token_mask: torch.Tensor | None = None  # (total_tokens,) bool,图像 token 位
    runner_type: str = "generation"                # "generation" | "pooling"

_context = Context()

def get_context() -> Context:
    return _context

def reset_context():
    global _context
    _context = Context()

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, positions_3d=None, image_token_mask=None, runner_type="generation"):
    global _context
    _context = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, positions_3d, image_token_mask, runner_type)
