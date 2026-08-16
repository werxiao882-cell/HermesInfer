"""TP-aware 权重加载器(Qwen3-VL-Embedding-2B)。

为什么与生成路径的 load_weights_from_checkpoint 分开:生成路径的加载器绕过
linear.py 挂在 param 上的 weight_loader callable(它用 torch.cat + param.data.copy_
(full)),在 world_size>1 下会把 rank-0 的切片装到每个 rank 上(见 design §1.2、
R-7)。VL 路径必须调用 param.weight_loader(param, weight[, id]),让
ColumnParallelLinear / QKVColumnParallelLinear 按 tp_rank 正确切头、
VocabParallelEmbedding 分片词表。复制部分(vision tower / merger / deepstack / RMSNorm
/ EmbeddingHead)直接 copy 全量。

HF→custom 名 remapping:checkpoint 是 transformers Qwen3VLForConditionalGeneration 格式,
常见前缀是 model.visual.* 与 model.language_model.*(或文本部分直接 model.*)。对每个
权重试多种 remap,加载第一个能解析到 custom 参数的。R-2:确切 HF 名布局须对照下载的
checkpoint 核对;本实现试常见模式,未匹配的在摘要里列出供跟进。在 GPU 上用
tests/test_parity_qwen.py 验证。
"""
import torch
from torch import nn
import os
import re
from safetensors import safe_open


def default_weight_loader(param, weight):
    """形状一致直接 copy;用于无 weight_loader 的复制参数。"""
    if param.shape != weight.shape:
        raise ValueError(f"Shape mismatch: param {param.shape} vs weight {weight.shape}")
    param.data.copy_(weight)


_RE_LAYER = re.compile(r"layers\.(\d+)")

# HF 名 → custom 名 的后缀替换表。本 VL 模型用了短属性名(qkv/o/down/in_ln/post_ln/
# n1/n2/mlp_fc1/fc1/dsm),HF checkpoint 用长名(qkv_projection/o_proj/down_proj/
# input_layernorm/post_attention_layernorm/norm1/norm2/mlp.linear_fc1/merger.linear_fc1/
# deepstack_merger_list)。顺序敏感:先做带 .mlp. 的,再做 merger 的,最后通用。
_REMAP = [
    (".self_attn.qkv_projection.", ".self_attn.qkv."),   # 文本 decoder QKV(若直查出现)
    (".self_attn.o_proj.", ".self_attn.o."),             # 文本 o_proj
    (".mlp.down_proj.", ".mlp.down."),                   # 文本 down_proj
    (".input_layernorm.", ".in_ln."),                   # 文本 input_layernorm
    (".post_attention_layernorm.", ".post_ln."),        # 文本 post_attention_layernorm
    (".mlp.linear_fc1.", ".mlp_fc1."),                  # ViT block MLP(fc1)
    (".mlp.linear_fc2.", ".mlp_fc2."),                 # ViT block MLP(fc2)
    (".norm1.", ".n1."),                                # ViT block norm1
    (".norm2.", ".n2."),                                # ViT block norm2
    ("merger.linear_fc1.", "merger.fc1."),              # merger fc1
    ("merger.linear_fc2.", "merger.fc2."),              # merger fc2
    ("deepstack_merger_list", "dsm"),                   # deepstack mergers
]


def _remap(name: str) -> str:
    for a, b in _REMAP:
        name = name.replace(a, b)
    return name


def _candidate_custom_names(hf_name: str):
    """对一个 HF 权重名,剥前缀(model.*/model.language_model.*)后,应用 _REMAP 把
    HF 长名映射到 custom 短名。先 yield remap 后的名(命中率高),再 yield 原名兜底。"""
    n = hf_name
    for prefix in ("model.language_model.", "language_model.", "model."):
        if n.startswith(prefix):
            n = n[len(prefix):]
            break
    if n.startswith("visual."):
        yield _remap(n)   # 视觉部分需要 norm1→n1 等映射
        yield n
    yield _remap(n)       # 文本部分需要 o_proj→o 等映射
    yield n


def _load_param(model, custom_name, weight, merged_id=None):
    """加载一个权重到 custom 参数:有 weight_loader 就调它(TP 切片),
    否则 default_weight_loader 复制。merged_id 给 QKV('q'/'k'/'v')与 gate_up(0/1)用,
    让 QKVColumnParallelLinear 按 GQA 头边界写到合并 param 的正确槽。返回是否命中。"""
    try:
        param = model.get_parameter(custom_name)
    except (AttributeError, KeyError):
        return False
    if hasattr(param, "weight_loader"):
        if merged_id is not None:
            param.weight_loader(param, weight, merged_id)
        else:
            param.weight_loader(param, weight)
    else:
        default_weight_loader(param, weight)
    return True


def load_weights_vl(model: nn.Module, model_name_or_path: str):
    """VL 权重加载主入口。步骤:
    1) 下载/定位 checkpoint,收齐 safetensors 张量
    2) 文本 decoder QKV 合并(q+k+v)+ 按 'q'/'k'/'v' 走 weight_loader(GQA 感关切头)
    3) gate_up 合并(gate+up)+ 按 0/1 走 weight_loader
    4) 其余按名 remap 直拷或 weight_loader(visual/merger/deepstack/norm/embed_tokens)
    未匹配的记入 skipped,摘要里列出供 R-2 跟进。"""
    from huggingface_hub import snapshot_download

    checkpoint_path = model_name_or_path
    if not (checkpoint_path.startswith('~') or os.path.isdir(checkpoint_path)):
        checkpoint_path = snapshot_download(
            repo_id=model_name_or_path,
            allow_patterns=["*.safetensors", "*.json"],
            ignore_patterns=["*.msgpack", "*.h5", "*.bin"],
        )

    safetensor_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.safetensors')]
    if not safetensor_files:
        raise ValueError(f"No .safetensors files found in {checkpoint_path}")

    # 收齐所有 HF 权重(按 layer 分组方便后续 merge)
    hf_weights = {}
    for file in sorted(safetensor_files):
        with safe_open(os.path.join(checkpoint_path, file), framework='pt', device='cpu') as f:
            for k in f.keys():
                hf_weights[k] = f.get_tensor(k)

    loaded = set()
    skipped = []

    # 1) 文本 decoder QKV 合并(q_proj + k_proj + v_proj -> qkv_projection),TP-aware
    for hf_name, weight in list(hf_weights.items()):
        if not hf_name.endswith('.q_proj.weight'):
            continue
        m = _RE_LAYER.search(hf_name)
        if not m:
            continue
        layer = m.group(1)
        k_name = hf_name.replace('q_proj', 'k_proj')
        v_name = hf_name.replace('q_proj', 'v_proj')
        if k_name in hf_weights and v_name in hf_weights:
            # weight_loader 契约(linear.py:173):loaded_weights 是"单个分量"(q/k/v 各自),
            # 按 load_weight_id 选 Q/K/V 槽 + 按 tp_rank 切头。故传各分量,不能传 cat 后的 merged。
            q_w, k_w, v_w = weight, hf_weights[k_name], hf_weights[v_name]
            base = hf_name[:-len('.q_proj.weight')]  # e.g. ...layers.0.self_attn
            for cand in _candidate_custom_names(base):  # 剥 model.* 前缀
                custom = cand + ".qkv.weight"   # 模型属性是 qkv(短名),非 qkv_projection
                # 三个分量各调一次,id 'q'/'k'/'v' 让 weight_loader 写到合并 param 的对应槽
                if _load_param(model, custom, q_w, merged_id="q") and \
                   _load_param(model, custom, k_w, merged_id="k") and \
                   _load_param(model, custom, v_w, merged_id="v"):
                    loaded.update({hf_name, k_name, v_name})
                    break

    # 2) gate_up 合并(gate_proj + up_proj -> gate_up),TP-aware
    for hf_name, weight in list(hf_weights.items()):
        if not hf_name.endswith('.gate_proj.weight'):
            continue
        m = _RE_LAYER.search(hf_name)
        if not m:
            continue
        up_name = hf_name.replace('gate_proj', 'up_proj')
        if up_name in hf_weights:
            # 同理:gate/up 各自单独传,id 0/1
            g_w, u_w = weight, hf_weights[up_name]
            base = hf_name[:-len('.gate_proj.weight')]
            for cand in _candidate_custom_names(base):
                custom = cand + ".mlp.gate_up.weight"
                if _load_param(model, custom, g_w, merged_id=0) and \
                   _load_param(model, custom, u_w, merged_id=1):
                    loaded.update({hf_name, up_name})
                    break

    # 3) 其余:按名 remap 直拷或 weight_loader(visual/merger/deepstack/norm/embed_tokens)
    for hf_name, weight in hf_weights.items():
        if hf_name in loaded:
            continue
        if any(x in hf_name for x in ('.k_proj.', '.v_proj.', '.up_proj.')):
            continue  # 已合并
        ok = False
        for cand in _candidate_custom_names(hf_name):
            if _load_param(model, cand, weight):
                ok = True
                break
        if not ok:
            skipped.append(hf_name)

    # 加载摘要:命中数 + 未匹配清单(供 R-2 跟进)
    print(f"\n{'='*70}\nVL Weight Loading Summary\n{'='*70}")
    print(f"loaded parameter groups: {len(loaded)}")
    if skipped:
        print(f"unmatched HF weights ({len(skipped)}): inspect name remapping (R-2)")
        for s in skipped[:15]:
            print(f"  - {s}")
        if len(skipped) > 15:
            print(f"  ... and {len(skipped) - 15} more")
    print(f"{'='*70}")
    return loaded
