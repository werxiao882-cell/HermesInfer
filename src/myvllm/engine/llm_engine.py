import atexit
import torch.distributed as dist
import time
import torch.multiprocessing as mp

from myvllm.engine.sequence import Sequence
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.model_runner import ModelRunner
from myvllm.sampling_parameters import SamplingParams
from transformers import AutoTokenizer


def worker_process(config, rank, event):
    """Worker process function that initializes ModelRunner and enters loop."""
    # FIRST print before any other code
    import sys
    import os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # Line buffering
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    model_runner = ModelRunner(config, rank, event)
    model_runner.loop()


class LLMEngine:
    def __init__(self, config: dict):
        self.config = config
        world_size = config.get("world_size", 1)
        ctx = mp.get_context("spawn")
        self.processes = []
        self.events = []
        for i in range(1, world_size):
            event = ctx.Event()
            process = ctx.Process(target=worker_process, args=(config, i, event))
            self.events.append(event)
            self.processes.append(process)
            process.start()
        # start the engine only on the master thread with rank = 0
        self.model_runner = ModelRunner(config, rank=0, event=self.events)
        # runner_type:"generation"(因果 LM,默认) | "pooling"(embedding,纯 prefill)
        self.runner_type = config.get("runner_type", "generation")
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name_or_path", "gpt2"))
        # VL/pooling 路径用 AutoProcessor(chat template + <|image_pad|> 插入 + 像素抽取)
        self.processor = None
        if self.runner_type == "pooling":
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(config.get("model_name_or_path"))

        # scheduler 需在 model_runner 之后初始化:world_size>1 时 ModelRunner.__init__
        # 调 dist.init_process_group() 这是个 collective barrier —— rank-0 会阻塞到所有
        # worker rank 都加入。scheduler 应在此 rendezvous 完成后创建。
        # world_size==1 时无 barrier,无真实依赖。
        # pooling 模式额外传 max_image_patches(vision tower 的 OOM 守卫预算)
        mm = config.get("multimodal", {})
        self.scheduler = Scheduler(
            max_num_sequences=config.get("max_num_sequences", 16),
            max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
            max_cached_blocks=config.get("max_cached_blocks", 1024),
            block_size=config.get("block_size", 256),
            eos=config.get("eos", 50256),
            runner_type=self.runner_type,
            max_image_patches=mm.get("max_image_patches", None),
        )

        atexit.register(self.exit)


    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for process in self.processes:
            process.join()

    # call scheduler to schedule the next batch
    # return scheduled sequences and whether it is for prefilling
    # call model_runner.run() to run the model
    # call postprocessor to process the outputs and update sequences and update block manager
    def step(self) -> tuple[list[tuple[int, list[int]]], int, bool]:
        scheduled_sequences, is_prefill = self.scheduler.schedule()
        num_processed_tokens = 0
        if not scheduled_sequences:
            return [], num_processed_tokens, is_prefill
        # run the model
        outputs = self.model_runner.call("run", scheduled_sequences, is_prefill)
        # Move outputs to CPU and convert them to a list
        if outputs is not None:
            outputs = outputs.cpu().tolist()
        # postprocess the outputs
        self.scheduler.postprocess(scheduled_sequences, outputs)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_sequences if seq.is_finished]
        num_processed_tokens = sum(len(seq) for seq in scheduled_sequences) if is_prefill else len(scheduled_sequences)

        return outputs, num_processed_tokens, is_prefill


    # add prompt string to the waiting queue by first transforming it to Sequence object
    def add_prompt(self, prompt: str, sampling_params: SamplingParams) -> None:
        self.scheduler.add_sequence(Sequence(token_ids=self.tokenizer.encode(prompt), block_size=self.config['block_size'],sampling_params=sampling_params))

    # given a list of prompts
    # add_prompt for each prompt
    # call step until all sequences are finished
    # return the generated texts
    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
        for prompt in prompts:
            self.add_prompt(prompt, sampling_params)
        generated_tokens = {}
        while not self.scheduler.is_finished():
            start_t = time.time()
            outputs, num_processed_tokens, is_prefill = self.step()
            end_t = time.time()
            running_time = end_t - start_t + 1e-10
            if is_prefill:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during prefilling")
            else:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during decoding")
            generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})

        generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(generated_tokens.keys())]
        output = {'text': [self.tokenizer.decode(tokens) for tokens in generated_tokens], 'token_ids': generated_tokens}
        return output

    # ---- pooling / embedding path (VL) ----

    def _add_input(self, item: dict, instruction: str, seq_id_counter):
        """把一条 {text,image} 请求经 AutoProcessor 转成 Sequence(带 mm_data)入队。
        build_multimodal_inputs 产出 input_ids + mm_token_type_ids + pixel_values +
        image_grid_thw + image_token_spans,挂到 Sequence.mm_data。"""
        from myvllm.models.qwen3_vl import build_multimodal_inputs, MultimodalData
        built = build_multimodal_inputs(self.processor, instruction, item)
        token_ids = built["input_ids"].tolist()
        seq = Sequence(token_ids=token_ids, block_size=self.config['block_size'])
        seq.mm_data = MultimodalData(
            pixel_values=built["pixel_values"],
            grid_thw=built["grid_thw"],
            token_types=built["token_types"],
            image_token_spans=built["image_token_spans"],
        )
        self.scheduler.add_sequence(seq)
        return seq.seq_id

    def encode(self, inputs: list[dict], pooling: dict | None = None) -> list:
        """把 文本/图像/图像+文本 输入编码为归一化向量。

        每条 item:{"text": str} | {"image": 路径} | {"text": str, "image": 路径},
        可选 per-item "instruction"(默认 "Represent the user's input.",instruction-aware)。
        返回按输入顺序的 (embed_dim,) CPU 张量列表。

        流程:逐条 _add_input 入 waiting → 循环 schedule()+run() 直到全部完成 →
        按输入顺序收集 embedding。Scheduler 自动按 token/image-patch 预算分批 prefill。"""
        if self.runner_type != "pooling":
            raise RuntimeError("encode() requires runner_type='pooling'")
        default_instr = self.config.get("pooling", {}).get(
            "instruction", "Represent the user's input.")
        ids = []
        for item in inputs:
            instr = item.get("instruction", default_instr)
            ids.append(self._add_input(item, instr, Sequence.counter))

        embeddings = {}
        while not self.scheduler.is_finished():
            seqs, is_prefill = self.scheduler.schedule()
            if not seqs:
                break  # 无 waiting 即完成
            # run() 在 pooling 模式返回 (num_seqs, dim) embedding(rank 0)
            emb = self.model_runner.call("run", seqs, True)
            self.scheduler.postprocess(seqs, emb)  # 标记 FINISHED、移出 running
            if emb is not None:
                emb_cpu = emb.detach().cpu()
                for seq, e in zip(seqs, emb_cpu):
                    embeddings[seq.seq_id] = e
        # 恢复输入顺序(schedule 分批会打乱,但 seq_id 唯一,按 ids 顺序取)
        return [embeddings[i] for i in ids]
