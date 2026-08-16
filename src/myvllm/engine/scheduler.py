from collections import deque
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.block_manager import BlockManager


class Scheduler:
    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int, max_cached_blocks: int, block_size: int, eos: int, runner_type: str = "generation", max_image_patches: int | None = None):
        # block manager
        self.block_manager = BlockManager(max_cached_blocks, block_size)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_sequences = max_num_sequences
        # sequence queue
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.eos = eos
        self.runner_type = runner_type
        self.max_image_patches = max_image_patches  # OOM guard for the vision tower


    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0

    def add_sequence(self, sequence: Sequence):
        if self.runner_type == "pooling":
            # pooling 无 KV cache:不做 num_blocks 容量校验,直接入 waiting。
            # 容量约束在 _schedule_pooling 里用 token / image-patch 预算兜。
            self.waiting.append(sequence)
            return
        # Reject up front what the block manager could never satisfy, otherwise the
        # sequence sits in `waiting` forever and only surfaces as a stalled engine.
        capacity = len(self.block_manager.blocks)
        if sequence.num_blocks > capacity:
            raise ValueError(
                f"Sequence {sequence.seq_id} needs {sequence.num_blocks} blocks "
                f"({len(sequence)} tokens at block_size={self.block_manager.block_size}) "
                f"but the KV cache only holds {capacity}. "
                f"Raise max_cached_blocks or block_size, or shorten the prompt."
            )
        self.waiting.append(sequence)


    def _seq_image_patches(self, seq: Sequence) -> int:
        """估算一条序列的图像 patch 总数(经 spatial_merge 缩后),用于 OOM 预算。
        grid_thw.prod(-1)//sms² = 每图 merged 后 token 数;多图求和。无图返回 0。"""
        mm = getattr(seq, "mm_data", None)
        if mm is None or mm.grid_thw is None:
            return 0
        return int((mm.grid_thw.prod(-1) // (getattr(self, "_sms", 2) ** 2)).sum())


    def schedule(self) -> tuple[list[Sequence], bool]:
        # pooling 模式走纯 prefill 调度,绕开 block_manager / decode / preempt
        if self.runner_type == "pooling":
            return self._schedule_pooling()
        scheduled_sequences = []
        current_scheduled_tokens = 0
        # An empty schedule is only legitimate when this call freed blocks by
        # preempting, so the next call can make progress. See the guard below.
        preempted = False
        # try schedule for prefilling from waiting queue if not exceeding limits
        while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
            seq = self.waiting[0]
            if self.block_manager.can_allocate(seq) and len(seq) + current_scheduled_tokens <= self.max_num_batched_tokens:
                seq = self.waiting.popleft() # remove from waiting
                self.block_manager.allocate(seq)
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += len(seq)
            else:
                break
        if scheduled_sequences:
            return scheduled_sequences, True

        # try schedule for completion from running queue
        while self.running:
            seq = self.running.popleft()
            # use can_append to check whether we can append one more token
            if not self.block_manager.can_append(seq):
                preempted = True
                if self.running:
                    self.running.appendleft(seq)
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                if current_scheduled_tokens >= self.max_num_batched_tokens or len(scheduled_sequences) >= self.max_num_sequences:
                    self.running.appendleft(seq)
                    break
                # append one token
                self.block_manager.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += 1 # only one token for completion

        # re-add to running queue in the same order
        if scheduled_sequences:
            self.running.extendleft(reversed(scheduled_sequences))
        elif not preempted and (self.waiting or self.running):
            # Nothing was scheduled and nothing was preempted, so no engine state
            # changed: every later schedule() would take the same decisions and
            # LLMEngine.generate() would spin forever. Fail loudly instead.
            raise RuntimeError(
                "Scheduler made no progress: "
                f"{len(self.waiting)} waiting and {len(self.running)} running sequences, "
                f"{len(self.block_manager.free_block_ids)} of "
                f"{len(self.block_manager.blocks)} blocks free. "
                "This means either a sequence that cannot fit in the KV cache, or "
                "blocks leaked because their ref_count never returned to 0."
            )

        return scheduled_sequences, False

    def _schedule_pooling(self) -> tuple[list[Sequence], bool]:
        """纯 prefill 调度:每条序列恰好在一步内完成(无 decode、无 preempt、
        不碰 block_manager)。按 token 预算 + 可选 image-patch 预算(vision tower 的
        OOM 守卫)分批。waiting 为空时返回 ([], False) 作为完成信号。

        返回 (seqs, is_prefill):is_prefill 恒 True(因为只有 prefill),调度空时
        返回 False 让 LLMEngine.encode 的循环终止。"""
        if not self.waiting:
            return [], False  # 完成:无 waiting 即终止编码循环
        scheduled = []
        cur_tokens = 0
        cur_patches = 0
        while self.waiting and len(scheduled) < self.max_num_sequences:
            seq = self.waiting[0]
            patches = self._seq_image_patches(seq)
            # 超出 token 预算:留到下一批(前提是本批已有内容,否则单条超长会死循环;
            # 此处简化:第一条总是放入,后续超预算就 break)
            if len(seq) + cur_tokens > self.max_num_batched_tokens and scheduled:
                break
            # 超出 image-patch 预算:留到下一批,防 vision tower 显存尖峰
            if self.max_image_patches is not None and cur_patches + patches > self.max_image_patches and scheduled:
                break
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            scheduled.append(seq)
            cur_tokens += len(seq)
            cur_patches += patches
        return scheduled, True if scheduled else False


    def preempt(self, seq: Sequence) -> None:
        # pooling 模式不可能走到 preempt(无 decode 无 KV 竞争);若被调到说明逻辑错误。
        if self.runner_type == "pooling":
            raise AssertionError("preempt() is unreachable in pooling mode")
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)


    # postprocess after generation to check whether sequences are finished
    # if finished, deallocate blocks
    def postprocess(self, seqs: list[Sequence], outputs) -> None:
        if self.runner_type == "pooling":
            # pooling:outputs 是 embedding 张量(num_seqs, dim),但这里不消费它
            # (LLMEngine.encode 直接从 run() 取结果)。postprocess 只负责把每条
            # 序列标记 FINISHED 并移出 running——无 token 追加、无 eos/max_tokens
            # 检查、无 KV 释放(根本没分配 KV)。
            for seq in seqs:
                seq.status = SequenceStatus.FINISHED
                if seq in self.running:
                    self.running.remove(seq)
            return
        for seq, token_id in zip(seqs, outputs):
            seq.append_token(token_id)
            # Check stopping conditions:
            # EOS token
            # Reached max_tokens limit (number of completion tokens)
            # Reached max_model_length limit (total sequence length including prompt)
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            stop_due_to_max_tokens = seq.num_completion_tokens >= seq.max_tokens
            stop_due_to_max_length = seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)