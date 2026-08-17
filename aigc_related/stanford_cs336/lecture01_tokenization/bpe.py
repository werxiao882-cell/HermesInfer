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
        """统计相邻byte的频率"""
        for i in range(len(tokens) - 1):
            new_token = tokens[i] + tokens[i+1]
            if new_token not in status:
                status[new_token] = 0
            status[new_token] += 1

    def _merge_pair_tokens(self, tokens, new_token):
        """合并列表中相邻的token"""
        merged_tokens = []
        i = 0
        while i< len(tokens):
            if i+1<len(tokens) and tokens[i] + tokens[i+1] == new_token:
                merged_tokens.append(tokens[i]+tokens[i+1])
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1
        return merged_tokens

    def train(self, corpus_list, vocab_size):
        # 单字节是最基本的token，初始化词表
        for idx in range(256):
            self.bytes2id[bytes([idx])] = idx
        self.next_id = 256

        # 将输入的语料库的内容转化为字节
        corpus_bytes_list = []
        for sub_corpus in corpus_list:
            current_bytes = [bytes([b]) for b in sub_corpus.encode("utf-8")]
            corpus_bytes_list.append(current_bytes)
        
        progress = tqdm(total=vocab_size - 256, desc = "")
        while True:
            # 如果词汇表足够大的话，退出训练
            if self.next_id >= vocab_size:
                break
            
            # 统计相邻的token频率
            status = {}
            for tokens in corpus_bytes_list:
                self._count_pair_tokens(tokens, status)

            # 没有更多相邻token，无法生成更多token，退出训练
            if not status:
                break

            # 合并最高频率的相邻token，作为新的token加入词表
            new_token = max(status, key = status.get)

            new_tokens_list = []
            for tokens in corpus_bytes_list:
                new_tokens_list.append(self._merge_pair_tokens(tokens, new_token))
            corpus_bytes_list = new_tokens_list

            # new_token加入词表
            self.bytes2id[new_token] = self.next_id

            # 更新进度条
            self.next_id = self.next_id + 1
            progress.update(1)

        self.id2bytes = {v:k for k,v in self.bytes2id.items()}

    def vocab_size(self):
        return self.next_id

    def encode(self, text):
        "BPE编码"
        pattern='('+'|'.join([re.escape(token) for token in self.special_str2id])+')'
        splits = re.split(pattern, text)

        encode_idx, encode_tokens = [], []
        for sub_text in splits:
            if sub_text in self.special_str2id:
                encode_idx.append(self.special_str2id[sub_text])
                encode_tokens.append(sub_text.encode("utf-8"))
            else:
                tokens = [bytes([token]) for token in sub_text.encode("utf-8")]

                while True:
                    # 1. 统计当前所有相邻的pair
                    status = {}
                    self._count_pair_tokens(tokens, status)

                    # 2. 寻找已经存在于此表中并且优先级比较好的合并规则
                    new_token = None
                    for merge_token in status:
                        if merge_token in self.bytes2id and (new_token is None or self.bytes2id[merge_token] < self.bytes2id[new_token]):
                            new_token = merge_token

                    # 3. 没有需要合并的pair时，则退出
                    if new_token is None:
                        break

                    # 4. 合并pair
                    tokens = self._merge_pair_tokens(tokens, new_token)
                
                encode_idx.extend([self.bytes2id[token] for token in tokens])
                encode_tokens.extend(tokens)
        return encode_idx, encode_tokens

    def decode(self, ids):
        bytes_list = []
        for id in ids:
            if id in self.special_id2str:
                bytes_list.append(self.special_id2str[id].encode("utf-8"))
            else:
                bytes_list.append(self.id2bytes[id])
        return b''.join(bytes_list).decode('utf-8', errors="replace")

    def add_special_tokens(self, special_tokens):
        """添加特殊token"""
        for token in special_tokens:
            if token not in self.special_str2id:
                self.special_str2id[token] = self.next_id
                self.special_id2str[self.next_id] = token
                self.next_id = self.next_id + 1

    def save(self, save_path):
        """保存为bin文件"""
        with open(save_path,'wb') as fp:
            fp.write(pickle.dumps((self.bytes2id,self.special_str2id,self.next_id)))

    def load(self, file_path):
        """加载bin文件"""
        with open(file_path, "rb") as fp:
            self.bytes2id, self.special_str2id, self.next_id = pickle.loads(fp.read())
        self.id2bytes = {v:k for k,v in self.bytes2id.items()}
        self.special_id2str = {v:k for k,v in self.special_str2id.items()}

if __name__ == "__main__":
    chinese_corpus = open("./dataset/train_chinese.txt", "r").read()
    english_corpus = open("./dataset/train_english.txt", "r").read()

    # Build tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(corpus_list = [chinese_corpus, english_corpus], vocab_size = 5000)
    tokenizer.add_special_tokens((['<|im_start|>','<|im_end|>','<|endoftext|>','<|padding|>']))
    tokenizer.save("./tokenizer.bin")

    # Test
    tokenizer = BPETokenizer()
    tokenizer.load("./tokenizer.bin")
    print("=> vocab_size:", tokenizer.vocab_size())

    # Encode
    idx, tokens = tokenizer.encode("<|im_start|>system\nyou are a helper assistant\n<|im_end|>\n<|im_start|>user\n今天的天气\n<|im_end|><|im_start|>assistant\n")
    print("=> encoded_result:", idx, tokens)

    # Decode
    decoded_result = tokenizer.decode(idx)
    print("=> decoded_result:", decoded_result)