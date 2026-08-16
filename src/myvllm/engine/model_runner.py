import math
import torch
import pickle
import torch.distributed as dist
from pathlib import Path
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.models.llama import LlamaForCausalLM
from myvllm.layers.sampler import SamplerLayer
from myvllm.engine.sequence import Sequence
from myvllm.utils import *

class ModelRunner:
    def __init__(self, config: dict, rank: int, event: Event | list[Event]):
        self.config = config
        self.event = event

        # set distributed config
        self.block_size = config['block_size']
        self.world_size = config['world_size']
        self.enforce_eager = config.get('enforce_eager', False)
        # runner_type:"generation"(因果 LM,默认) | "pooling"(embedding,纯 prefill)
        # 决定是否走 KV cache / CUDA graph / 采样,以及用哪条加载器。
        self.runner_type = config.get('runner_type', 'generation')

        self.rank = rank
        dist.init_process_group('nccl', "tcp://localhost:12345", world_size=config['world_size'], rank=rank)
        torch.cuda.set_device(rank)

        # set model
        path_str = self.config['model_name_or_path']
        model_name = Path(path_str).name
        match model_name:
            case 'Qwen3-0.6B':
                self.model = Qwen3ForCausalLM(
                    vocab_size=config['vocab_size'],
                    hidden_size=config['hidden_size'],
                    num_heads=config['num_heads'],
                    head_dim=config['head_dim'],
                    scale=config['scale'],
                    num_kv_heads=config['num_kv_heads'],
                    rms_norm_epsilon=config['rms_norm_epsilon'],
                    qkv_bias=config['qkv_bias'],
                    base=config['base'],
                    max_position=config['max_position'],
                    intermediate_size=config['intermediate_size'],
                    ffn_bias=config['ffn_bias'],
                    num_layers=config['num_layers'],
                    tie_word_embeddings=config['tie_word_embeddings'],
                    block_size=self.block_size,
                )
            case 'Llama-3.2-1B-Instruct':
                self.model = LlamaForCausalLM(
                    vocab_size=config['vocab_size'],
                    hidden_size=config['hidden_size'],
                    head_dim=config['head_dim'],
                    num_qo_heads=config['num_qo_heads'],
                    num_kv_heads=config['num_kv_heads'],
                    has_attn_bias=config['has_attn_bias'],
                    rms_norm_epsilon=config['rms_norm_epsilon'],
                    rope_base=config['rope_base'],
                    max_position_embeddings=config['max_position_embeddings'],
                    intermediate_size=config['intermediate_size'],
                    ffn_bias=config['ffn_bias'],
                    num_layers=config['num_layers'],
                    block_size=self.block_size,
                    tie_word_embeddings=config['tie_word_embeddings'],
                )
            case 'Qwen3-VL-Embedding-2B':
                # 构建 VL embedding 模型:复制 vision tower + TP 分片 28 层文本 decoder
                # + 复制 EmbeddingHead。配置默认值对齐 Qwen3-VL-Embedding-2B 的 config.json
                # (dim 2048 / heads 16 / kv 8 / layers 28 / mrope [24,20,20] / deepstack [5,11,17])。
                from myvllm.models.qwen3_vl import Qwen3VLForEmbedding
                pcfg = config.get('pooling', {})
                self.model = Qwen3VLForEmbedding(
                    vocab_size=config.get('vocab_size', 151936),
                    hidden_size=config.get('hidden_size', 2048),
                    num_heads=config.get('num_heads', 16),
                    head_dim=config.get('head_dim', 128),
                    num_kv_heads=config.get('num_kv_heads', 8),
                    intermediate_size=config.get('intermediate_size', 6144),
                    num_layers=config.get('num_layers', 28),
                    rms_norm_epsilon=config.get('rms_norm_epsilon', 1e-6),
                    base=config.get('base', 5_000_000),
                    mrope_section=config.get('mrope_section', [24, 20, 20]),
                    block_size=self.block_size,
                    tie_word_embeddings=config.get('tie_word_embeddings', True),
                    vision_depth=config.get('vision_depth', 24),
                    vision_hidden_size=config.get('vision_hidden_size', 1024),
                    vision_intermediate_size=config.get('vision_intermediate_size', 4096),
                    vision_num_heads=config.get('vision_num_heads', 16),
                    patch_size=config.get('patch_size', 16),
                    temporal_patch_size=config.get('temporal_patch_size', 2),
                    in_channels=config.get('in_channels', 3),
                    out_hidden_size=config.get('out_hidden_size', 2048),
                    spatial_merge_size=config.get('spatial_merge_size', 2),
                    num_position_embeddings=config.get('num_position_embeddings', 2304),
                    deepstack_visual_indexes=config.get('deepstack_visual_indexes', [5, 11, 17]),
                    pooling_mode=pcfg.get('mode', 'last_token'),
                    normalize=pcfg.get('normalize', True),
                    mrl_dim=pcfg.get('mrl_dim', None),
                )
            case _:
                raise Exception(f"Unsupported model: {config['model_name_or_path']}")

        # Load weights in GPU (model moved to GPU before loading weights)
        self.model = self.model.cuda(rank)

        # Load pretrained weights if model_name_or_path is provided
        if config.get('model_name_or_path'):
            if self.runner_type == 'pooling':
                # VL 走 TP-aware 加载器:调 per-param weight_loader 按 tp_rank 正确切头
                # (修 R-7,生成路径不动)。复制部分(vision/deepstack/EmbeddingHead)全量。
                from myvllm.utils.loader_vl import load_weights_vl
                load_weights_vl(self.model, config['model_name_or_path'])
            else:
                from myvllm.utils.loader import load_weights_from_checkpoint
                load_weights_from_checkpoint(self.model, config['model_name_or_path'])

        # Load weights in CPU (move the model to GPU after loading weights)
        # self.model = self.model.cuda(rank)

        self.sampler = SamplerLayer()

        # Store default dtype before it's needed in allocate_kv_cache
        self.default_dtype = torch.get_default_dtype()

        # Debug flag for first decode step
        self._first_decode = False

        # warm up model so that we know peak memory usage
        # allocate kv cache
        # capture cuda graph for decoding
        # 这三者都是生成路径(decode KV cache、图重放)的关切;pooling 是纯 prefill,
        # 无 decode/KV cache/CUDA graph,故整体跳过,把显存让给 vision 激活。
        # 若不跳过,warmup_model 的文本形 forward 与 allocate_kv_cache 会报错/浪费。
        if self.runner_type == 'generation':
            self.warmup_model()
            self.allocate_kv_cache()
            if not self.enforce_eager:
                self.capture_cudagraph()

        torch.set_default_device(f'cuda:{rank}')
        torch.set_default_dtype(self.default_dtype)

        # IMPORTANT: Set up shared memory and barrier AFTER all model initialization
        # This ensures both ranks complete warmup/allocation before rank 1 enters its event loop
        if self.world_size > 1:
            # Synchronize before setting up shared memory
            dist.barrier()
            if self.rank == 0:
                # Try to clean up existing shared memory first
                try:
                    old_shm = SharedMemory(name='myvllm')
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass  # Doesn't exist, which is fine
                self.shm = SharedMemory(name='myvllm', create=True, size=2**20)
                # Barrier to ensure rank 1 waits until shared memory is created
                dist.barrier()
            else:
                # Wait for rank 0 to create shared memory
                dist.barrier()
                self.shm = SharedMemory(name='myvllm')
                # Don't call self.loop() here - let the spawning code handle it
                # Otherwise we'll be stuck in an infinite loop during __init__

    # only use read when rank != 0
    def read_shm(self):
        assert self.world_size > 1 and self.rank != 0, "read_shm can only be called when world_size > 1 and rank != 0"
        self.event.wait()
        n = int.from_bytes(self.shm.buf[:4], 'little') # read length
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    # only use write when rank == 0
    def write_shm(self, method_name: str, args: tuple):
        assert self.world_size > 1 and self.rank == 0, "write_shm can only be called when world_size > 1 and rank == 0"
        # encode the length first
        # Flatten: (method_name, args) where args is a tuple -> (method_name, *args)
        data = pickle.dumps((method_name, *args))
        n = len(data)
        self.shm.buf[:4] = n.to_bytes(4, 'little')
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    # close shared memory, destroy process group, delete graphs
    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs
            del self.graph_vars
        torch.cuda.synchronize()
        # Check if process group exists before destroying
        if dist.is_initialized():
            dist.destroy_process_group()
    
    # wait to read method and args from shared memory
    # execute the method with args
    # write results back to shared memory
    def loop(self):
        assert self.world_size > 1 and self.rank != 0, "loop can only be called when world_size > 1 and rank != 0"
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args) # Unpack args when calling
            if method_name == 'exit':
                self.exit()
                break

    # will be called by both rank == 0 and rank != 0
    # given method name and args from shared memory
    # execute the method and return results
    def call(self, method_name: str, *args: dict):
        if self.world_size > 1 and self.rank == 0: # will be called in main engine
            self.write_shm(method_name, args)
        method = getattr(self, method_name, None)
        if method:
            return method(*args)
        raise ValueError(f"Unknown method: {method_name}")

    # cleanup memory
    # compute max number of sequence based on max token and max model length
    # run empty sequence to warm up the model
    # clear memory
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_tokens = self.config['max_num_batch_tokens']
        max_model_length = self.config['max_model_length']
        batch_size = max_tokens // max_model_length
        seqs = [Sequence(token_ids=[0]*max_model_length, block_size=self.config['block_size']) for _ in range(batch_size)]
        self.run(seqs, is_prefill=True)
        torch.cuda.empty_cache()

    # allocate kv cache memory blocks for model
    def allocate_kv_cache(self):
        # find all available memory
        free_mem, total_mem = torch.cuda.mem_get_info()
        total_free_mem = free_mem * self.config['gpu_memory_utilization']
        peak_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.peak']
        current_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.current']
        # reserve some room for peak memory usage during model execution
        available_mem = total_free_mem - (peak_mem_usage - current_mem_usage)
        
        # find parameters to compute kv cache size
        num_layers = self.config['num_layers']
        num_kv_heads = self.config['num_kv_heads'] // self.world_size
        head_dim = self.config['head_dim'] if 'head_dim' in self.config else self.config['hidden_size'] // self.config['num_heads']

        # check whether the current free memory can hold at least one block
        # compute the actual byte required of each block
        block_bytes = self.block_size * 2 * num_layers * num_kv_heads * head_dim * self.default_dtype.itemsize
        num_available_kv_blocks = int(available_mem // block_bytes)
        assert num_available_kv_blocks >= 1, f'Not enough memory to hold at least one block of KV cache on rank {self.rank}'
        
        # Synchronize max_cached_blocks across all ranks.
        # Each rank independently computed num_available_kv_blocks from its own
        # free GPU memory. Ranks may differ slightly: rank-0 carries extra overhead
        # (NCCL buffers, process-group state) so it often has less free memory than
        # workers. Without sync, the scheduler (which runs only on rank-0) would use
        # rank-0's local value and could allocate more blocks than some rank can hold,
        # causing an OOM on that rank during KV cache writes.
        if self.world_size > 1:
            print(f"[Rank {self.rank}] Local max_cached_blocks: {num_available_kv_blocks}")
            per_rank_max_blocks_tensor = torch.tensor(
                num_available_kv_blocks,
                dtype=torch.long,
                device=f'cuda:{self.rank}'
            )
            # all_reduce with MIN: every rank learns the most conservative limit,
            # i.e. the block count that even the most memory-constrained rank can serve.
            # This single agreed-upon value is then stored in config so the Scheduler
            # (initialized afterwards on rank-0) never allocates more blocks than any
            # rank can physically hold.
            dist.all_reduce(per_rank_max_blocks_tensor, op=dist.ReduceOp.MIN)
            self.config['max_cached_blocks'] = per_rank_max_blocks_tensor.item()
        else:
            # Single GPU: no cross-rank sync needed; use the local value directly.
            self.config['max_cached_blocks'] = num_available_kv_blocks
        if self.rank == 0:
            print(f"[Rank 0] Global max_cached_blocks (min): {self.config['max_cached_blocks']}")

        # allocate max possible kv cache for the model, instead for each sequence
        # this is the key for paged attention: one giant KV cache pool, divided into blocks
        # IMPORTANT: Use zeros() instead of empty() to avoid garbage values
        allocated_kv_cache = torch.zeros(2, self.config['num_layers'], self.config['max_cached_blocks'], self.block_size, num_kv_heads, head_dim, device=f'cuda:{self.rank}')
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                module.k_cache = allocated_kv_cache[0, layer_id]
                module.v_cache = allocated_kv_cache[1, layer_id]
                layer_id += 1

    # given seqs
    # prepare the data needed for a prefill forward pass
    # taking prefix cache into consideration: 
    # input_ids, positions, cu_seqlens_q/k, slot_mapping (where to write new KV values), block_tables (where to read KV values)
    # cu_seqlens_q = [0, 3, 5, 9]
    #               │  │  │  │
    #               │  │  │  └─ end of seq3 (position 9)
    #               │  │  └──── end of seq2 (position 5)
    #               │  └─────── end of seq1 (position 3)
    #               └────────── start (position 0)
    def prepare_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        # length: sum of all input_ids after prefix cache
        input_ids = []
        # length: sum of all input_ids after prefix cache
        slot_mappings = []
        # length: num_seqs
        seqlens_q = []
        # length: num_seqs
        seqlens_k = []
        # length: num_seqs + 1
        cu_seqlens_q = [0]
        # length: num_seqs + 1
        cu_seqlens_k = [0]
        # block_tables: num_seqs x num_blocks (padded)
        block_tables = []
        for seq in seqs:
            token_ids = seq.token_ids
            num_cached_tokens = seq.num_cached_tokens
            input_ids.extend(token_ids[num_cached_tokens:])
            seqlens_q.append(len(token_ids) - num_cached_tokens)
            seqlens_k.append(len(token_ids))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q[-1])
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k[-1])
            if seq.block_table:
                for i, block_id in enumerate(seq.block_table[seq.num_cached_blocks:]):
                    if seq.num_cached_blocks + i != seq.num_blocks - 1:
                        slot_mappings.extend(list(range(block_id * self.block_size, (block_id+1) * self.block_size)))
                    else:
                        slot_mappings.extend(list(range(block_id * self.block_size, block_id * self.block_size + seq.last_block_num_tokens)))
        if cu_seqlens_q[-1] < cu_seqlens_k[-1]:
            # pad block_tables
            all_block_tables = [seq.block_table for seq in seqs]
            max_num_blocks = max(len(bt) for bt in all_block_tables)
            for i, seq in enumerate(seqs):
                block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
                block_tables.append(block_table)
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)

        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            slot_mapping=slot_mapping_tensor,
            context_lens=None,
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
        )
        return input_ids


    # prepare input data for decoding
    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        input_ids = []
        context_lens = []   
        slot_mappings = []  
        block_tables = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            context_lens.append(len(seq))
            slot_mappings.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        all_block_tables = [seq.block_table for seq in seqs]
        max_num_blocks = max(len(bt) for bt in all_block_tables)
        for i, seq in enumerate(seqs):
            block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
            block_tables.append(block_table)
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            context_lens=torch.tensor(context_lens, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
        )
        return input_ids    

    # prepare the temperature
    def prepare_sample(self, seqs: list[Sequence]) -> None:
        return torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    # when prefilling, directly compute model forward + logits
    # when decoding, use cuda graph execution to speed up
    # allocate input_ids, positions, slot_mapping, context_lens, block_tables, outputs
    # into graph_variable, and then replay the graph
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill or self.enforce_eager:
            # For varlen prefill, keep input_ids as 1D (concatenated tokens)
            # Do NOT unsqueeze - flash_attn_varlen_func expects 1D input with cu_seqlens
            hidden_states = self.model(input_ids)
            logits = self.model.compute_logits(hidden_states)
        else:
            bs = input_ids.size(0)
            context = get_context()

            # finds smallest captured graph that fits the batch size
            graph = self.graphs[next(bs_ for bs_ in self.graphs.keys() if bs_ >= bs)]
            vars = self.graph_vars
            # copy input data into graph variables
            vars['input_ids'][:bs].copy_(input_ids)
            vars['slot_mapping'][:bs].fill_(-1)
            vars['slot_mapping'][:bs].copy_(context.slot_mapping)
            vars["context_lens"].zero_()
            vars['context_lens'][:bs].copy_(context.context_lens)
            vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # replay the graph
            graph.replay()
            logits = self.model.compute_logits(vars['outputs'][:bs])

        return logits


    # prepare prefill
    # prepare sample
    # run model
    # sample logits
    # reset context
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # pooling 模式直接走 _run_pooling(无 decode、无采样、无 cuda graph)
        if self.runner_type == 'pooling':
            return self._run_pooling(seqs)
        if is_prefill:
            input_ids = self.prepare_prefill(seqs)
        else:
            input_ids = self.prepare_decode(seqs)
        logits = self.run_model(input_ids, is_prefill)
        # only sample when rank == 0
        token_ids = None
        if self.rank == 0:
            token_ids = self.sampler(logits, self.prepare_sample(seqs))
        reset_context()
        return token_ids

    @torch.inference_mode()
    def _run_pooling(self, seqs: list[Sequence]):
        """Pooling(embedding)执行路径:纯 prefill,跑 vision tower + VL 文本 decoder
        + pooling head,返回 (num_seqs, embed_dim)。无 KV cache、无 CUDA graph、无采样。
        与生成路径 run() 对应,但绕开 decode/cuda graph/sampler(那些在 __init__ 里
        runner_type=='pooling' 时已整体跳过分配)。"""
        input_ids, vl_inputs = self._prepare_prefill_vl(seqs)
        # 模型 forward 自己跑 vision tower(复制)+ TP 分片文本 decoder + EmbeddingHead
        emb = self.model(input_ids, **vl_inputs)
        reset_context()
        # EmbeddingHead 复制(残差流在每 rank 都是 full),故每 rank 产出相同 embedding;
        # 仅 rank 0 返回(对标生成路径的 rank-0 采样约定,见 run())。
        return emb if self.rank == 0 else None

    def _prepare_prefill_vl(self, seqs: list[Sequence]):
        """构造一次 pooling prefill 的全部输入。步骤:
        1) 打包 input_ids + cu_seqlens_q(无前缀缓存,pooling 不复用 KV)
        2) 收集每条序列的 mm_data + MRoPE 位置输入(token_types、image_grids)
        3) compute_mrope_positions 算 (3, total_tokens) 的 T/H/W 位置
        4) 拼批级 pixel_values/grid_thw,并把每图 image_token_spans 加偏移映射进 packed 序列
        5) 构造 image_token_mask(图像 token 的 bool 掩码,供 deepstack scatter)
        6) set_context(slot_mapping=None 让 Attention 跳过 KV 存储;positions_3d/image_token_mask 入 context)
        返回 (input_ids_tensor, {pixel_values, grid_thw, image_token_spans, cu_seqlens_q})。"""
        from myvllm.models.qwen3_vl import compute_mrope_positions
        # ---- 1) 打包 input_ids + cu_seqlens_q,顺带收 mm_data ----
        input_ids = []
        cu_seqlens_q = [0]
        per_seq_mm = []
        per_seq_pos_inputs = []
        sms = self.config.get('spatial_merge_size', 2)
        for seq in seqs:
            ids = seq.token_ids
            input_ids.extend(ids)
            cu_seqlens_q.append(cu_seqlens_q[-1] + len(ids))
            mm = getattr(seq, "mm_data", None)
            per_seq_mm.append(mm)
            # 给 compute_mrope_positions 准备每条序列的输入
            if mm is not None and mm.token_types is not None:
                per_seq_pos_inputs.append({
                    "input_ids": torch.tensor(ids, device='cpu', dtype=torch.long),
                    "token_types": mm.token_types.to('cpu'),
                    "image_grids": [tuple(int(x) for x in g) for g in mm.grid_thw.tolist()] if mm.grid_thw is not None else [],
                })
            else:
                # 纯文本:token_types 全 0(文本),无 image_grids
                per_seq_pos_inputs.append({
                    "input_ids": torch.tensor(ids, device='cpu', dtype=torch.long),
                    "token_types": torch.zeros(len(ids), dtype=torch.int),
                    "image_grids": [],
                })

        # ---- 2) MRoPE 3D 位置(T/H/W)按批打包,每条序列 current_pos 从 0 起 ----
        # VL 模型始终用 MRoPE:文本 token 三轴 T=H=W=arange(_positions_single 处理),
        # 图像 token 从 patch 网格派生。故即便纯文本批也要算 positions_3d —— 之前用
        # `if any(image_grids)` 守卫会让纯文本请求 positions_3d=None,模型 forward 无条件
        # 调 rotary → NoneType 下标崩溃(README 明确支持纯文本 query,必修)。
        positions_3d = compute_mrope_positions(per_seq_pos_inputs, spatial_merge_size=sms)
        positions_3d = positions_3d.cuda(self.rank)

        # ---- 3) vision tower 输入:批级拼 pixel_values/grid_thw,并把每图 spans 加偏移 ----
        # 注意:vision tower 由 model.forward 自己跑(复制),这里只收集输入;
        # image_token_spans 需从"每序列局部坐标"转成"packed 批全局坐标"。
        pixel_values = None
        grid_thw = None
        image_token_spans = []
        has_image = any(m is not None and m.has_image for m in per_seq_mm)
        if has_image:
            pv_list, gt_list = [], []
            offset = 0  # 当前序列在 packed input_ids 里的起点
            for seq, mm in zip(seqs, per_seq_mm):
                ids_len = len(seq.token_ids)
                if mm is not None and mm.has_image:
                    pv_list.append(mm.pixel_values)
                    gt_list.append(mm.grid_thw)
                    # 把该序列的 (s,e) 加 offset 变全局坐标
                    for (s, e) in mm.image_token_spans:
                        image_token_spans.append((offset + s, offset + e))
                offset += ids_len
            pixel_values = torch.cat(pv_list, dim=0).cuda(self.rank)
            grid_thw = torch.cat(gt_list, dim=0).cuda(self.rank)

        # ---- 4) image_token_mask(total_tokens 的 bool),供 deepstack 在图像 token 位置 scatter ----
        total = len(input_ids)
        if image_token_spans:
            mask = torch.zeros(total, dtype=torch.bool)
            for (s, e) in image_token_spans:
                mask[s:e] = True
            image_token_mask = mask.cuda(self.rank)
        else:
            image_token_mask = None

        # ---- 5) set_context:slot_mapping=None → Attention 跳过 KV 存(polling 无 cache);
        #        positions_3d / image_token_mask 经 context 单例传给模型与 attention ----
        cu_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_q,           # pooling 下 Q=K(同一序列),用同一组 cu_seqlens
            max_seqlen_q=max(cu_seqlens_q[i+1]-cu_seqlens_q[i] for i in range(len(cu_seqlens_q)-1)),
            max_seqlen_k=0,
            slot_mapping=None,           # 关键:让 Attention.forward 跳过 store_kvcache 分支
            context_lens=None,
            block_tables=None,
            positions_3d=positions_3d,
            image_token_mask=image_token_mask,
            runner_type='pooling',
        )

        input_ids_t = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        # 这些会作为 model.forward 的关键字参数(pixel_values/grid_thw/image_token_spans/cu_seqlens_q)
        return input_ids_t, {
            "pixel_values": pixel_values,
            "grid_thw": grid_thw,
            "image_token_spans": image_token_spans,
            "cu_seqlens_q": cu_q,
        }

    # capture the CUDA graph:
    # pre-allocation at maximum sizes: allocated onece and reuse for all graphs
    # capture for different common batch sizes: [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    # with torch.cuda.graph(graph, self.graph_pool):
    #        run model() and exact sequence of CUDA kernels for running self.model() will be captured
    # (later use graph.replay() to run the captured graph)
    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        max_bs = self.config['max_num_seqs']
        max_len = self.config['max_model_length']
        max_num_blocks = math.ceil(max_len / self.block_size)
        # for decoding, input is always of shape (batch_size, 1)
        input_ids = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # for paged attention
        # where to write new KV values in the cache
        slot_mapping = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # how many tokens each sequence has processed
        context_lens = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # where to read KV values in the cache
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=f'cuda:{self.rank}')
        # output logits
        outputs = torch.zeros(max_bs, self.config['vocab_size'], device=f'cuda:{self.rank}')

        # graphs to be captured for different batch sizes
        batch_sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        graph_pool = None

        for batch_size in reversed(batch_sizes):
            graph = torch.cuda.CUDAGraph()
            set_context(
                is_prefill=False,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=slot_mapping[:batch_size],
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
            )
            outputs[:batch_size] = self.model(input_ids[:batch_size])

            with torch.cuda.graph(graph, graph_pool):
                outputs[:batch_size] = self.model(input_ids[:batch_size])
                if graph_pool is None:
                    graph_pool = graph.pool()
            # store the captured graph
            self.graphs[batch_size] = graph

            # make sure that the capture is done before resetting and next capture
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )