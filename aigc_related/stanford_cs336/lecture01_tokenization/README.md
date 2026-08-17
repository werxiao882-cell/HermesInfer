# Tokenizer 详解

## 目录
- [什么是 Tokenizer？](#什么是-tokenizer)
- [为什么需要 Tokenizer？](#为什么需要-tokenizer)
- [自然语言处理中的 Tokenizer 类型](#自然语言处理中的-tokenizer-类型)
  - [字符级 Tokenizer (Character-level)](#1-字符级-tokenizer-character-level)
  - [词级 Tokenizer (Word-level)](#2-词级-tokenizer-word-level)
  - [子词级 Tokenizer (Subword-level)](#3-子词级-tokenizer-subword-level)
- [BPE Tokenizer 详解](#bpe-tokenizer-详解)
  - [训练流程](#训练流程)
  - [编码流程](#编码流程-encode)
  - [解码流程](#解码流程-decode)
  - [代码实现](#代码实现)
- [特殊 Token 处理](#特殊-token-处理)

---

## 什么是 Tokenizer？

**Tokenizer**（分词器）是连接人类语言和机器学习模型的桥梁，它将原始文本（raw text）转换为模型能够理解的数字序列。在模型的输入和输出阶段中发挥着至关重要的作用：

### 模型输入（编码 Encode）阶段

1. **分词（Tokenize）**

   将文本拆分为词元（Token），常见的分词方式包括字符级、词级、子词级（如 BPE、WordPiece）等。

   ```
   输入: "你好世界"
   分词: ["你", "好", "世", "界"]
   ```

2. **映射（Mapping）**

   将每个词元映射为词汇表中的唯一 ID，生成的数字序列即为模型的输入。

   ```
   分词: ["你", "好", "世", "界"]
   映射: [1001, 1002, 1003, 1004]
   ```

### 模型输出（解码 Decode）阶段

1. **反映射（De-mapping）**

   模型输出的数字序列通过词汇表映射回对应的词元，二者是一一对应的关系。

   ```
   输出: [1001, 1002, 1003, 1004]
   反映射: ["你", "好", "世", "界"]
   ```

2. **文本重组**

   将解码后的词元以某种规则重新拼接为完整文本。

   ```
   反映射: ["你", "好", "世", "界"]
   重组: "你好世界"
   ```

---

## 为什么需要 Tokenizer？

### 1. **模型只能处理数字**

神经网络模型（如 Transformer、BERT、GPT）本质上是数学运算的集合，只能处理数字矩阵，无法直接理解文本。Tokenizer 将文本转换为数字序列，使得模型能够进行计算。

### 2. **降低词汇表大小**

如果为每个不同的词都分配一个 ID，词汇表会非常庞大（尤其是中文、日文等语言），导致：
- **内存开销巨大**：词汇表越大，Embedding 层的参数越多
- **稀疏性问题**：很多罕见词在训练数据中出现次数极少，难以学到有效表示
- **未登录词（OOV）问题**：遇到训练时未见过的词无法处理

**子词级 Tokenizer**（如 BPE）通过拆分成更小的单元，可以在较小的词汇表下表示几乎所有可能的文本。

### 3. **统一表示不同语言**

字节级 BPE 可以处理任何语言的文本（包括表情符号、特殊字符），因为所有文本最终都可以用 UTF-8 字节表示。

### 4. **提高模型泛化能力**

通过子词分割，模型可以学习词根、前缀、后缀等语言结构，更好地理解词汇的组成，从而对未见过的词也能进行合理的推理。

---

## 自然语言处理中的 Tokenizer 类型

### 1. 字符级 Tokenizer (Character-level)

**原理**：将文本拆分为单个字符。

**示例**：
```
输入: "hello"
分词: ["h", "e", "l", "l", "o"]
```

**优点**：
- 词汇表非常小（英文仅需 26 个字母 + 标点符号）
- 没有 OOV 问题（所有文本都能表示）
- 适合处理拼写错误、罕见词

**缺点**：
- 序列长度过长，模型计算开销大
- 字符本身缺乏语义信息，模型需要学习如何组合字符形成有意义的单元
- 不适合中文等语言（字符数量庞大）

**应用场景**：字符级语言模型、拼写检查

---

### 2. 词级 Tokenizer (Word-level)

**原理**：将文本按照空格、标点符号拆分为完整的词。

**示例**：
```
输入: "Hello, world!"
分词: ["Hello", ",", "world", "!"]
```

**优点**：
- 每个 token 具有完整的语义
- 序列长度较短
- 直观易懂

**缺点**：
- 词汇表庞大（英文需要几十万个词，中文更多）
- OOV 问题严重（遇到未见过的词无法处理）
- 对于形态丰富的语言（如德语、芬兰语）词汇表会爆炸
- 无法处理拼写错误

**应用场景**：早期的 NLP 模型（如 Word2Vec、GloVe）

---

### 3. 子词级 Tokenizer (Subword-level)

子词级 Tokenizer 是目前主流大模型（GPT、BERT、LLaMA）的选择，它在词级和字符级之间取得平衡。

#### 3.1 BPE (Byte Pair Encoding)

**原理**：从字符（或字节）开始，迭代地合并最频繁出现的相邻 token 对，直到达到目标词汇表大小。

**代表模型**：GPT-2、GPT-3、GPT-4、LLaMA、Qwen

**优点**：
- 词汇表大小可控
- 几乎没有 OOV 问题
- 可以学习常见词根、前缀、后缀
- 字节级 BPE 可以处理任何语言

**缺点**：
- 同一个词在不同上下文中可能被分割成不同的 token
- 训练需要大量语料

**详见下文 [BPE Tokenizer 详解](#bpe-tokenizer-详解)**

#### 3.2 WordPiece

**原理**：与 BPE 类似，但合并时选择能最大化训练数据似然的 token 对（而非频率最高）。

**代表模型**：BERT、DistilBERT

**优点**：
- 理论上比 BPE 更优（基于似然而非频率）
- 词汇表大小可控

**缺点**：
- 训练复杂度较高
- 需要大量语料

**示例**：
```
输入: "playing"
分词: ["play", "##ing"]  (## 表示子词)
```
---

## BPE Tokenizer 详解

### 什么是 BPE？

**BPE (Byte Pair Encoding)** 最初是一种数据压缩算法，后来被引入到 NLP 领域用于分词。现代大模型（如 GPT 系列）广泛使用 **字节级 BPE**，它在字节层面进行操作，可以处理任何 UTF-8 编码的文本。

### 核心思想

1. **初始化**：将文本拆分为最小单元（字节，对应 0-255 共 256 个 token）
2. **迭代合并**：统计所有相邻 token 对的频率，将最高频的 token 对合并为一个新 token
3. **重复**：重复步骤 2，直到词汇表达到目标大小

---

### 训练流程

#### 输入
- **语料库**（Corpus）：大量的文本数据（如维基百科、书籍、网页等）
- **目标词汇表大小**（Vocab Size）：如 50,000 或 100,000

#### 步骤

**Step 1: 初始化词汇表**

将所有文本转换为 UTF-8 字节序列，初始词汇表包含 256 个基础字节（0-255）。

```python
for idx in range(256):
    self.bytes2id[bytes([idx])] = idx
self.next_id = 256
```

**Step 2: 统计相邻 token 对的频率**

扫描整个语料库，统计所有相邻 token 对出现的次数。

```python
def _count_pair_tokens(self, tokens, status):
    """统计相邻 token 的频率"""
    for i in range(len(tokens) - 1):
        new_token = tokens[i] + tokens[i+1]
        if new_token not in status:
            status[new_token] = 0
        status[new_token] += 1
```

**示例**：
```
语料: "hello" + "hella"
字节序列: [b'h', b'e', b'l', b'l', b'o'] + [b'h', b'e', b'l', b'l', b'a']

相邻对频率:
b'he': 2
b'el': 2
b'll': 2
b'lo': 1
b'la': 1
```

**Step 3: 合并最高频的 token 对**

选择频率最高的 token 对，将其合并为一个新 token，并加入词汇表。

```python
# 合并最高频的 token 对
new_token = max(status, key=status.get)  # 假设是 b'he'

# 更新语料库中的所有匹配
def _merge_pair_tokens(self, tokens, new_token):
    """合并列表中相邻的 token"""
    merged_tokens = []
    i = 0
    while i < len(tokens):
        if i+1 < len(tokens) and tokens[i] + tokens[i+1] == new_token:
            merged_tokens.append(tokens[i] + tokens[i+1])
            i += 2
        else:
            merged_tokens.append(tokens[i])
            i += 1
    return merged_tokens
```

**示例**：
```
合并 b'he' 后:
[b'h', b'e', b'l', b'l', b'o'] -> [b'he', b'l', b'l', b'o']
[b'h', b'e', b'l', b'l', b'a'] -> [b'he', b'l', b'l', b'a']

词汇表: {0: b'h', 1: b'e', ..., 256: b'he'}
```

**Step 4: 重复迭代**

重复步骤 2 和 3，每次合并最高频的 token 对，直到词汇表达到目标大小。

```python
while self.next_id < vocab_size:
    # 统计频率
    status = {}
    for tokens in corpus_bytes_list:
        self._count_pair_tokens(tokens, status)
    
    if not status:
        break
    
    # 合并最高频对
    new_token = max(status, key=status.get)
    self.bytes2id[new_token] = self.next_id
    self.next_id += 1
    
    # 更新语料库
    new_tokens_list = []
    for tokens in corpus_bytes_list:
        new_tokens_list.append(self._merge_pair_tokens(tokens, new_token))
    corpus_bytes_list = new_tokens_list
```

**训练结果**：

- **词汇表**（bytes2id）：字节序列 → ID 的映射
- **反向词汇表**（id2bytes）：ID → 字节序列的映射

---

### 编码流程 (Encode)

编码是将原始文本转换为 token ID 序列的过程。

#### 输入
- **原始文本**（如 "hello world"）

#### 步骤

**Step 1: 处理特殊 Token**

如果文本中包含特殊 token（如 `<|im_start|>`、`<|endoftext|>`），需要先将其分离出来。

```python
# 使用正则表达式分割特殊 token
pattern = '(' + '|'.join([re.escape(token) for token in self.special_str2id]) + ')'
splits = re.split(pattern, text)

# 示例
text = "<|im_start|>hello<|im_end|>"
splits = ["<|im_start|>", "hello", "<|im_end|>"]
```

**Step 2: 逐段编码**

对于每个文本片段：
- 如果是特殊 token，直接映射为对应的 ID
- 如果是普通文本，进行 BPE 编码

```python
for sub_text in splits:
    if sub_text in self.special_str2id:
        # 特殊 token 直接映射
        encode_idx.append(self.special_str2id[sub_text])
    else:
        # 普通文本进行 BPE 编码
        tokens = [bytes([b]) for b in sub_text.encode("utf-8")]
        # ... BPE 合并逻辑
```

**Step 3: BPE 合并**

从单字节开始，迭代地查找词汇表中存在的、优先级最高的合并规则（ID 越小优先级越高），直到无法继续合并。

```python
while True:
    # 1. 统计当前所有相邻的 pair
    status = {}
    self._count_pair_tokens(tokens, status)
    
    # 2. 寻找已存在于词汇表中并且优先级最高的合并规则
    new_token = None
    for merge_token in status:
        if merge_token in self.bytes2id and (new_token is None or self.bytes2id[merge_token] < self.bytes2id[new_token]):
            new_token = merge_token
    
    # 3. 没有需要合并的 pair 时，退出
    if new_token is None:
        break
    
    # 4. 合并 pair
    tokens = self._merge_pair_tokens(tokens, new_token)
```

**为什么选择 ID 最小的？**

因为 ID 越小，说明这个 token 在训练时越早被创建，代表其在语料库中频率越高，应该优先合并。

**Step 4: 映射为 ID**

将最终的 token 序列映射为 ID 序列。

```python
encode_idx.extend([self.bytes2id[token] for token in tokens])
```

#### 示例

```python
tokenizer = BPETokenizer()
tokenizer.load("./tokenizer.bin")

text = "<|im_start|>hello<|im_end|>"
ids, tokens = tokenizer.encode(text)

# 输出:
# ids: [1000, 256, 257, 258, 1001]
# tokens: [b'<|im_start|>', b'he', b'll', b'o', b'<|im_end|>']
```

---

### 解码流程 (Decode)

解码是将 token ID 序列转换回原始文本的过程。

#### 输入
- **ID 序列**（如 `[1000, 256, 257, 258, 1001]`）

#### 步骤

**Step 1: 反映射**

将每个 ID 映射回对应的字节序列（或特殊 token 字符串）。

```python
bytes_list = []
for id in ids:
    if id in self.special_id2str:
        # 特殊 token
        bytes_list.append(self.special_id2str[id].encode("utf-8"))
    else:
        # 普通 token
        bytes_list.append(self.id2bytes[id])
```

**Step 2: 字节拼接**

将所有字节序列拼接成一个完整的字节流。

```python
full_bytes = b''.join(bytes_list)
```

**Step 3: UTF-8 解码**

将字节流解码为 UTF-8 字符串。

```python
text = full_bytes.decode('utf-8', errors="replace")
```

`errors="replace"` 表示如果遇到无效的 UTF-8 字节，将其替换为 `�`（而不是抛出异常）。

#### 示例

```python
ids = [1000, 256, 257, 258, 1001]
text = tokenizer.decode(ids)

# 输出: "<|im_start|>hello<|im_end|>"
```

---

### 代码实现

以下是完整的 BPE Tokenizer 实现（基于您提供的代码）：

```python
import pickle
from collections import OrderedDict
import re
from tqdm import tqdm

class BPETokenizer:
    def __init__(self):
        self.bytes2id = OrderedDict()
        self.id2bytes = OrderedDict()
        self.next_id = 0
        self.special_str2id = {}
        self.special_id2str = {}

    def _count_pair_tokens(self, tokens, status):
        """统计相邻 byte 的频率"""
        for i in range(len(tokens) - 1):
            new_token = tokens[i] + tokens[i+1]
            if new_token not in status:
                status[new_token] = 0
            status[new_token] += 1

    def _merge_pair_tokens(self, tokens, new_token):
        """合并列表中相邻的 token"""
        merged_tokens = []
        i = 0
        while i < len(tokens):
            if i+1 < len(tokens) and tokens[i] + tokens[i+1] == new_token:
                merged_tokens.append(tokens[i] + tokens[i+1])
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1
        return merged_tokens

    def train(self, corpus_list, vocab_size):
        """训练 BPE Tokenizer"""
        # 初始化词汇表：256 个基础字节
        for idx in range(256):
            self.bytes2id[bytes([idx])] = idx
        self.next_id = 256

        # 将语料库转换为字节序列
        corpus_bytes_list = []
        for sub_corpus in corpus_list:
            current_bytes = [bytes([b]) for b in sub_corpus.encode("utf-8")]
            corpus_bytes_list.append(current_bytes)
        
        progress = tqdm(total=vocab_size - 256, desc="Training BPE")
        
        while self.next_id < vocab_size:
            # 统计相邻 token 频率
            status = {}
            for tokens in corpus_bytes_list:
                self._count_pair_tokens(tokens, status)

            # 没有更多相邻 token
            if not status:
                break

            # 合并最高频率的相邻 token
            new_token = max(status, key=status.get)

            # 更新语料库
            new_tokens_list = []
            for tokens in corpus_bytes_list:
                new_tokens_list.append(self._merge_pair_tokens(tokens, new_token))
            corpus_bytes_list = new_tokens_list

            # 加入词汇表
            self.bytes2id[new_token] = self.next_id
            self.next_id += 1
            progress.update(1)

        self.id2bytes = {v: k for k, v in self.bytes2id.items()}

    def encode(self, text):
        """BPE 编码"""
        # 分离特殊 token
        pattern = '(' + '|'.join([re.escape(token) for token in self.special_str2id]) + ')'
        splits = re.split(pattern, text)

        encode_idx, encode_tokens = [], []
        for sub_text in splits:
            if sub_text in self.special_str2id:
                # 特殊 token
                encode_idx.append(self.special_str2id[sub_text])
                encode_tokens.append(sub_text.encode("utf-8"))
            else:
                # 普通文本：从字节开始迭代合并
                tokens = [bytes([b]) for b in sub_text.encode("utf-8")]

                while True:
                    status = {}
                    self._count_pair_tokens(tokens, status)

                    # 寻找词汇表中存在的、优先级最高的合并规则
                    new_token = None
                    for merge_token in status:
                        if merge_token in self.bytes2id and (new_token is None or self.bytes2id[merge_token] < self.bytes2id[new_token]):
                            new_token = merge_token

                    if new_token is None:
                        break

                    tokens = self._merge_pair_tokens(tokens, new_token)
                
                encode_idx.extend([self.bytes2id[token] for token in tokens])
                encode_tokens.extend(tokens)
        
        return encode_idx, encode_tokens

    def decode(self, ids):
        """BPE 解码"""
        bytes_list = []
        for id in ids:
            if id in self.special_id2str:
                bytes_list.append(self.special_id2str[id].encode("utf-8"))
            else:
                bytes_list.append(self.id2bytes[id])
        return b''.join(bytes_list).decode('utf-8', errors="replace")

    def add_special_tokens(self, special_tokens):
        """添加特殊 token"""
        for token in special_tokens:
            if token not in self.special_str2id:
                self.special_str2id[token] = self.next_id
                self.special_id2str[self.next_id] = token
                self.next_id += 1

    def save(self, save_path):
        """保存为 bin 文件"""
        with open(save_path, 'wb') as fp:
            fp.write(pickle.dumps((self.bytes2id, self.special_str2id, self.next_id)))

    def load(self, file_path):
        """加载 bin 文件"""
        with open(file_path, "rb") as fp:
            self.bytes2id, self.special_str2id, self.next_id = pickle.loads(fp.read())
        self.id2bytes = {v: k for k, v in self.bytes2id.items()}
        self.special_id2str = {v: k for k, v in self.special_str2id.items()}
```

#### 使用示例

```python
# 1. 训练 Tokenizer
tokenizer = BPETokenizer()
tokenizer.train(
    corpus_list=["Hello world", "你好世界"], 
    vocab_size=1000
)
tokenizer.add_special_tokens(['<|im_start|>', '<|im_end|>', '<|endoftext|>'])
tokenizer.save("./tokenizer.bin")

# 2. 加载 Tokenizer
tokenizer = BPETokenizer()
tokenizer.load("./tokenizer.bin")
print("词汇表大小:", tokenizer.vocab_size())

# 3. 编码
text = "<|im_start|>你好世界<|im_end|>"
ids, tokens = tokenizer.encode(text)
print("编码结果:", ids)
print("Token 序列:", tokens)

# 4. 解码
decoded_text = tokenizer.decode(ids)
print("解码结果:", decoded_text)
```

---

## 特殊 Token 处理

在实际应用中，Tokenizer 需要处理一些特殊的控制符号，如：

| 特殊 Token | 作用 |
|-----------|------|
| `<|endoftext|>` | 标记文档结束（GPT 模型） |
| `<|im_start|>` | 标记对话开始（Qwen、ChatGPT 格式） |
| `<|im_end|>` | 标记对话结束 |
| `<|padding|>` | 填充 token（对齐序列长度） |
| `<unk>` | 未知 token（词级 tokenizer 中使用） |
| `<s>` / `</s>` | 句子开始/结束（LLaMA 模型） |

### 为什么需要特殊 Token？

1. **结构化信息**：标记对话轮次、角色（system/user/assistant）
2. **训练信号**：告诉模型哪里是文档边界、哪里需要生成结束符
3. **批处理**：填充 token 用于对齐不同长度的序列

### 实现方式

特殊 token 在训练时不参与 BPE 合并，而是直接分配固定的 ID：

```python
tokenizer.add_special_tokens(['<|im_start|>', '<|im_end|>', '<|endoftext|>'])
```

在编码时，先用正则表达式提取特殊 token，再对普通文本进行 BPE 编码：

```python
pattern = '(' + '|'.join([re.escape(token) for token in self.special_str2id]) + ')'
splits = re.split(pattern, text)
```

---

## BPE 的优势与局限

### 优势

1. **通用性强**：字节级 BPE 可以处理任何 UTF-8 文本（多语言、emoji、代码）
2. **无 OOV 问题**：任何未见过的词都可以分解为已知的子词或字节
3. **压缩效率高**：常见词用一个 token 表示，罕见词拆分为多个子词
4. **可控的词汇表大小**：通过调整 `vocab_size` 控制模型规模

### 局限

1. **贪心算法**：BPE 使用贪心策略（每次合并频率最高的对），不一定是全局最优
2. **对噪声敏感**：拼写错误、标点符号会导致分词不一致
3. **无法处理歧义**：同一个词在不同上下文中可能被分割成不同的 token
4. **训练依赖语料**：训练语料的质量和分布会显著影响 tokenizer 的性能

---

## 总结

| Tokenizer 类型 | 词汇表大小 | OOV 问题 | 序列长度 | 代表模型 |
|---------------|----------|---------|---------|---------|
| 字符级 | 很小 (~100) | 无 | 很长 | CharCNN |
| 词级 | 很大 (>100k) | 严重 | 短 | Word2Vec |
| BPE | 中等 (30k-100k) | 几乎无 | 中等 | GPT、LLaMA |
| WordPiece | 中等 (30k-100k) | 几乎无 | 中等 | BERT |

**BPE Tokenizer** 是目前最流行的分词方法，在效率、通用性和性能之间取得了良好的平衡，是理解现代大模型的关键基础组件。

---

## 参考资料

- [Hugging Face Tokenizers 文档](https://huggingface.co/docs/tokenizers/)
- [OpenAI Tokenizer 可视化工具](https://platform.openai.com/tokenizer)