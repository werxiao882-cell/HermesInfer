import torch.distributed as dist

_usp_group = None
_usp_rank = 0
_usp_world_size = 1


def init_usp_group(world_size, rank, port=12346):
    global _usp_group, _usp_rank, _usp_world_size
    _usp_world_size = world_size
    _usp_rank = rank

    if world_size == 1:
        _usp_group = None
        return

    if dist.is_initialized():
        _usp_group = dist.new_group(backend="nccl")
    else:
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{port}",
            world_size=world_size,
            rank=rank,
        )
        _usp_group = dist.group.WORLD


def get_usp_group():
    return _usp_group


def usp_world_size():
    return _usp_world_size


def usp_rank():
    return _usp_rank
