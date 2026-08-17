import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model:int, vocab_size:int) -> None:
        super(InputEmbedding, self).__init__()
        # NOTE：nn.Embdding在Pytorch中是torch.nn.Parameter来封装其底层的权重张量
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float) -> None:
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class LayerNormalization(nn.Module):
    def __init__(self, features:int, eps:float=1e-6) -> None:
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class ResidualConnection(nn.Module):
    def __init__(self, features:int, dropout:float) -> None:
        super(ResidualConnection, self).__init__()
        self.norm = LayerNormalization(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor, sublayer:nn.Module) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, dropout:float) -> None:
        super(MultiHeadAttentionBlock, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / math.sqrt(self.d_k)

    @staticmethod
    def scaled_dot_product_attention(query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask:torch.Tensor=None, dropout:nn.Module=None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) * (1.0 / math.sqrt(query.size(-1)))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(scores, dim=-1)
        if dropout is not None:
            attention_weights = dropout(attention_weights)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights

    def forward(self, q, k, v, mask=None) -> torch.Tensor:
        batch_size = q.size(0)

        query = self.linear_q(q)
        key = self.linear_k(k)
        value = self.linear_v(v)

        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.scaled_dot_product_attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.linear_out(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward: FeedForward, d_model:int, dropout:float) -> None:
        super(EncoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward
        self.residual_connection = nn.ModuleList([
            ResidualConnection(d_model, dropout),
            ResidualConnection(d_model, dropout)
        ])
    
    def forward(self, x:torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x:torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward: FeedForward, d_model:int, dropout:float) -> None:
        super(DecoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward
        self.residual_connection = nn.ModuleList([
            ResidualConnection(d_model, dropout),
            ResidualConnection(d_model, dropout),
            ResidualConnection(d_model, dropout)
        ])
    
    def forward(self, x:torch.Tensor, enc_output:torch.Tensor, src_mask:torch.Tensor=None, tgt_mask:torch.Tensor=None) -> torch.Tensor:
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, enc_output, enc_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x:torch.Tensor, enc_output:torch.Tensor, src_mask:torch.Tensor=None, tgt_mask:torch.Tensor=None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int) -> None:
        super(ProjectionLayer, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
class Transformer(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, src_embedding:InputEmbedding, tgt_embedding:InputEmbedding, src_positional_encoding:PositionalEncoding, tgt_positional_encoding:PositionalEncoding, projection:ProjectionLayer) -> None:
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_positional_encoding = src_positional_encoding
        self.tgt_positional_encoding = tgt_positional_encoding
        self.projection = projection
    
    def encode(self, src:torch.Tensor, src_mask:torch.Tensor=None) -> torch.Tensor:
        src_embedded = self.src_positional_encoding(self.src_embedding(src))
        return self.encoder(src_embedded, src_mask)

    def decode(self, tgt:torch.Tensor, enc_output:torch.Tensor, src_mask:torch.Tensor=None, tgt_mask:torch.Tensor=None) -> torch.Tensor:
        tgt_embedded = self.tgt_positional_encoding(self.tgt_embedding(tgt))
        return self.decoder(tgt_embedded, enc_output, src_mask, tgt_mask)

    def project(self, x:torch.Tensor) -> torch.Tensor:
        return self.projection(x) 

def build_transformer(src_vocab_size:int, tgt_vocab_size:int, seq_len_src:int, seq_len_tgt:int, d_model:int=512, d_ff:int=2048, num_heads:int=8, num_layers:int=6, dropout:float=0.1) -> Transformer:
    src_embedding = InputEmbedding(d_model, src_vocab_size)
    tgt_embedding = InputEmbedding(d_model, tgt_vocab_size)
    src_positional_encoding = PositionalEncoding(d_model, seq_len_src, dropout)
    tgt_positional_encoding = PositionalEncoding(d_model, seq_len_tgt, dropout)
    encoder_layers = nn.ModuleList([
        EncoderBlock(
            MultiHeadAttentionBlock(d_model, num_heads, dropout),
            FeedForward(d_model, d_ff, dropout),
            d_model,
            dropout
        ) for _ in range(num_layers)
    ])
    encoder = Encoder(encoder_layers)

    decoder_layers = nn.ModuleList([
        DecoderBlock(
            MultiHeadAttentionBlock(d_model, num_heads, dropout),
            MultiHeadAttentionBlock(d_model, num_heads, dropout),
            FeedForward(d_model, d_ff, dropout),
            d_model,
            dropout
        ) for _ in range(num_layers)
    ])
    decoder = Decoder(decoder_layers)

    projection = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embedding, tgt_embedding, src_positional_encoding, tgt_positional_encoding, projection)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer