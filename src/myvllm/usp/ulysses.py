import torch
import torch.distributed as dist
from .group import get_usp_group, usp_world_size


def all_to_all_seq2head(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Transpose from sequence-sharded to head-sharded layout.

    Input:  (seq//P, num_heads, head_dim)  — each rank holds 1/P of sequence, all heads
    Output: (num_heads//P, seq, head_dim)  — each rank holds all sequence, 1/P of heads

    Uses dist.all_to_all to redistribute data across ranks.
    """
    P = usp_world_size()
    if P == 1:
        return x.transpose(0, 1).contiguous()

    assert num_heads % P == 0, f"num_heads ({num_heads}) must be divisible by world_size ({P})"

    seq_per_rank, _, head_dim = x.shape
    local_heads = num_heads // P

    x = x.view(seq_per_rank, P, local_heads, head_dim)
    x = x.permute(1, 0, 2, 3).contiguous()

    output = torch.empty_like(x)
    dist.all_to_all_single(output, x, group=get_usp_group())

    output = output.view(P * seq_per_rank, local_heads, head_dim)
    output = output.transpose(0, 1).contiguous()

    return output


def all_to_all_head2seq(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Transpose from head-sharded back to sequence-sharded layout.

    Input:  (num_heads//P, seq, head_dim)  — each rank holds 1/P of heads, all sequence
    Output: (seq//P, num_heads, head_dim)  — each rank holds all heads, 1/P of sequence

    Inverse of all_to_all_seq2head.
    """
    P = usp_world_size()
    if P == 1:
        return x.transpose(0, 1).contiguous()

    assert num_heads % P == 0, f"num_heads ({num_heads}) must be divisible by world_size ({P})"

    local_heads, seq_full, head_dim = x.shape
    seq_per_rank = seq_full // P

    x = x.transpose(0, 1).contiguous()
    x = x.view(seq_per_rank, P, local_heads, head_dim)
    x = x.permute(1, 0, 2, 3).contiguous()

    output = torch.empty_like(x)
    dist.all_to_all_single(output, x, group=get_usp_group())

    output = output.view(seq_per_rank, num_heads, head_dim)

    return output
