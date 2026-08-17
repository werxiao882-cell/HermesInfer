import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math

def precompute_freqs_cis(dim_model: int, end: int = 2048, theta: float = 10000.0):
    # Llama: concat cos/sin to match [x_left, x_right] pairing
    freqs = 1.0 / (theta ** (torch.arange(0, dim_model, 2)[: (dim_model // 2)].float() / dim_model))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    # 生成 [cos0, cos1, ..., cos0, cos1, ...] 的形式 (前半部分和后半部分重复)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Llama 风格 RoPE，输入 shape: [batch, seq, n_heads, dim_rope]
    unsqueeze_dim=2 对应 n_heads 维度（在 transpose 之前调用）
    """
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    q_len = xq.shape[1]
    k_len = xk.shape[1]

    # freqs_cos/sin shape: [seq, dim_rope]
    # xq/xk shape: [batch, seq, n_heads, dim_rope]
    # 在 batch(0) 和 n_heads(unsqueeze_dim) 两个维度插入 1 以支持广播
    q_cos = freqs_cos[:q_len].unsqueeze(0).unsqueeze(unsqueeze_dim)  # [1, q_len, 1, dim_rope]
    q_sin = freqs_sin[:q_len].unsqueeze(0).unsqueeze(unsqueeze_dim)
    k_cos = freqs_cos[:k_len].unsqueeze(0).unsqueeze(unsqueeze_dim)  # [1, k_len, 1, dim_rope]
    k_sin = freqs_sin[:k_len].unsqueeze(0).unsqueeze(unsqueeze_dim)

    xq_out = (xq * q_cos) + (rotate_half(xq) * q_sin)
    xk_out = (xk * k_cos) + (rotate_half(xk) * k_sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class MultiHeadLatentAttention(nn.Module):
    """
        Multi-Head Latent Attention(MLA) Module As in DeepSeek_V2 pape
        Key innovation from standard MHA:
             1. Low-Rank Key-Value Joint Compression 
             2. Decoupled Rotary Position Embedding
             
    Args:
        dim_model:  Total dimension of the model.
        num_head: Number of attention heads.
        dim_kv:      K/V compression dimension
        dim_query:     Q compression dimension
        dim_rope: Dimension for Rotary Position Embedding
        dropout:  Dropout rate for attention scores.
        bias:     Whether to include bias in linear projections.

        dim_head:   Inferred from dim_model//num_head

    Inputs:
        sequence: input sequence for self-attention and the query for cross-attention
        key_value_state: input for the key, values for cross-attention
    """
    def __init__(
        self, 
        dim_model,             # Infer dim_head from dim_model
        num_head, 
        dim_kv, 
        dim_query, 
        dim_rope, 
        dropout=0.1, 
        bias=True,
        max_batch_size=32,   # For KV cache sizing
        max_seq_len=2048     # For KV cache sizing 
        ):
        super().__init__()
        
        assert dim_model % num_head == 0, "dim_model must be divisible by num_head"
        assert dim_kv < dim_model, "Compression dim should be smaller than model dim"
        assert dim_query < dim_model, "Query compression dim should be smaller than model dim"
        
        self.dim_model = dim_model
        self.num_head = num_head
        # Verify dimensions match up
        assert dim_model % num_head == 0, f"dim_model ({dim_model}) must be divisible by num_head ({num_head})"
        self.dim_head=dim_model//num_head
        self.dim_kv = dim_kv
        self.dim_query = dim_query
        self.dim_rope = dim_rope
        self.dropout_rate = dropout  # Store dropout rate separately

        # Linear down-projection(compression) transformations
        self.DKV_proj = nn.Linear(dim_model, dim_kv, bias=bias)
        self.DQ_proj = nn.Linear(dim_model, dim_query, bias=bias)
        
        # linear up-projection transformations
        self.UQ_proj = nn.Linear(dim_query, dim_model, bias=bias)
        self.UK_proj = nn.Linear(dim_kv, dim_model, bias=bias)
        self.UV_proj = nn.Linear(dim_kv, dim_model, bias=bias)

        # Linear RoPE-projection
        self.RQ_proj = nn.Linear(dim_query, num_head*dim_rope, bias=bias)
        self.RK_proj = nn.Linear(dim_model, dim_rope, bias=bias)
        
        # linear output transformations
        self.output_proj = nn.Linear(dim_model, dim_model, bias=bias)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Initiialize scaler
        self.scaler = float(1.0 / math.sqrt(self.dim_head + dim_rope)) # Store as float in initialization

        # Initialize C_KV and R_K cache for inference
        self.cache_kv = torch.zeros((max_batch_size, max_seq_len, dim_kv))
        self.cache_rk = torch.zeros((max_batch_size, max_seq_len, dim_rope))

        # Initialize freqs_cis for RoPE (Llama cos/sin style)
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(dim_rope, max_seq_len * 2)

    def forward(
        self, 
        sequence, 
        key_value_states = None, 
        att_mask=None,
        use_cache=False,
        start_pos: int = 0
    ):
        """
        Forward pass supporting both standard attention and cached inference
        Input shape: [batch_size, seq_len, dim_model=num_head * dim_head]
        Args:
            sequence: Input sequence [batch_size, seq_len, dim_model]
            key_value_states: Optional states for cross-attention
            att_mask: Optional attention mask
            use_cache: Whether to use KV caching (for inference)
            start_pos: Position in sequence when using KV cache
        """
        batch_size, seq_len, model_dim = sequence.size()
        # prepare for RoPE
        self.freqs_cos = self.freqs_cos.to(sequence.device)
        self.freqs_sin = self.freqs_sin.to(sequence.device)
        freqs_cos = self.freqs_cos[start_pos:]
        freqs_sin = self.freqs_sin[start_pos:]

        # Check only critical input dimensions
        assert model_dim == self.dim_model, f"Input dimension {model_dim} doesn't match model dimension {self.dim_model}"
        if key_value_states is not None:
            assert key_value_states.size(-1) == self.dim_model, \
            f"Cross attention key/value dimension {key_value_states.size(-1)} doesn't match model dimension {self.dim_model}"

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        # Determine kv_seq_len early
        kv_seq_len = key_value_states.size(1) if is_cross_attention else seq_len
        
        # Linear projections and reshape for multi-head, in the order of Q, K/V
        # Down and up projection for query
        C_Q = self.DQ_proj(sequence)     #[batch_size, seq_len, dim_query]
        Q_state = self.UQ_proj(C_Q)      #[batch_size, seq_len, dim_model]
        # Linear projection for query RoPE pathway
        Q_rotate = self.RQ_proj(C_Q)      #[batch_size, seq_len, num_head*dim_rope]

        if use_cache:
            #Equation (41) in DeepSeek-v2 paper: cache c^{KV}_t
            self.cache_kv = self.cache_kv.to(sequence.device)

            # Get current compressed KV states
            current_kv = self.DKV_proj(key_value_states if is_cross_attention else sequence) #[batch_size, kv_seq_len, dim_kv]
            # Update cache using kv_seq_len instead of seq_len
            self.cache_kv[:batch_size, start_pos:start_pos + kv_seq_len] = current_kv
            # Use cached compressed KV up to current position
            C_KV = self.cache_kv[:batch_size, :start_pos + kv_seq_len]

            #Equation (43) in DeepSeek-v2 paper: cache the RoPE pathwway for shared key k^R_t
            assert self.cache_rk.size(-1) == self.dim_rope, "RoPE cache dimension mismatch"
            self.cache_rk = self.cache_rk.to(sequence.device)
            # Get current RoPE key
            current_K_rotate = self.RK_proj(key_value_states if is_cross_attention else sequence) #[batch_size, kv_seq_len, dim_rope]
            # Update cache using kv_seq_len instead of seq_len
            self.cache_rk[:batch_size, start_pos:start_pos + kv_seq_len] = current_K_rotate
            # Use cached RoPE key up to current position
            K_rotate = self.cache_rk[:batch_size, :start_pos + kv_seq_len] #[batch_size, cached_len, dim_rope]

            """handling attention mask"""
            if att_mask is not None:
                # Get the original mask shape
                mask_size = att_mask.size(-1)
                cached_len = start_pos + kv_seq_len        # cached key_len, including previous key
                assert C_KV.size(1) == cached_len, \
            f"Cached key/value length {C_KV.size(1)} doesn't match theoretical length {cached_len}"
                
                # Create new mask matching attention matrix shape
                extended_mask = torch.zeros(
                    (batch_size, 1, seq_len, cached_len),  # [batch, head, query_len, key_len]
                    device=att_mask.device,
                    dtype=att_mask.dtype
                )
                
                # Fill in the mask appropriately - we need to be careful about the causality here
                # For each query position, it should only attend to cached positions up to that point
                for i in range(seq_len):
                    extended_mask[:, :, i, :(start_pos + i + 1)] = 0  # Can attend
                    extended_mask[:, :, i, (start_pos + i + 1):] = float('-inf')  # Cannot attend
                    
                att_mask = extended_mask
        else:
            # Compression projection for C_KV
            C_KV = self.DKV_proj(key_value_states if is_cross_attention else sequence) #[batch_size, kv_seq_len, dim_kv]\
            # RoPE pathway for *shared* key
            K_rotate = self.RK_proj(key_value_states if is_cross_attention else sequence)
            
        # Up projection for key and value
        K_state = self.UK_proj(C_KV)               #[batch_size, kv_seq_len/cached_len, dim_model]
        V_state = self.UV_proj(C_KV)               #[batch_size, kv_seq_len/cached_len, dim_model]

        Q_state = Q_state.view(batch_size, seq_len, self.num_head, self.dim_head)

        # After getting K_state from projection, get its actual sequence length
        actual_kv_len = K_state.size(1)    # kv_seq_len or start_pos + kv_seq_len
        # in cross-attention, key/value sequence length might be different from query sequence length
        # Use actual_kv_len instead of kv_seq_len for reshaping
        K_state = K_state.view(batch_size, actual_kv_len, self.num_head, self.dim_head) 
        V_state = V_state.view(batch_size, actual_kv_len, self.num_head, self.dim_head)

        #Apply RoPE to query and shared key
        Q_rotate = Q_rotate.view(batch_size, seq_len, self.num_head, self.dim_rope)
        K_rotate = K_rotate.unsqueeze(2).expand(-1, -1, self.num_head, -1)  # [batch, cached_len, num_head, dim_rope]
        Q_rotate, K_rotate = apply_rotary_emb(Q_rotate, K_rotate, freqs_cos=freqs_cos, freqs_sin=freqs_sin)

        # Concatenate along head dimension
        Q_state = torch.cat([Q_state, Q_rotate], dim=-1)  # [batch_size, seq_len, num_head, dim_head + dim_rope]
        K_state = torch.cat([K_state, K_rotate], dim=-1)  # [batch_size, actual_kv_len, num_head, dim_head + dim_rope]

        # Scale Q by 1/sqrt(dim_k)
        Q_state = Q_state * self.scaler
        Q_state = Q_state.transpose(1, 2)  # [batch_size, num_head, seq_len, head_dim]
        K_state = K_state.transpose(1, 2)  # [batch_size, num_head, actual_kv_len, head_dim]
        V_state = V_state.transpose(1, 2)  # [batch_size, num_head, actual_kv_len, head_dim]

        # Compute attention matrix: QK^T
        self.att_matrix = torch.matmul(Q_state, K_state.transpose(-1,-2)) 
    
        # apply attention mask to attention matrix
        if att_mask is not None and not isinstance(att_mask, torch.Tensor):
            raise TypeError("att_mask must be a torch.Tensor")

        if att_mask is not None:
            self.att_matrix = self.att_matrix + att_mask
        
        # apply softmax to the last dimension to get the attention score: softmax(QK^T)
        att_score = F.softmax(self.att_matrix, dim = -1)
    
        # apply drop out to attention score
        att_score = self.dropout(att_score)
    
        # get final output: softmax(QK^T)V
        att_output = torch.matmul(att_score, V_state)
        assert att_output.size(0) == batch_size, "Batch size mismatch"
        assert att_output.size(2) == seq_len, "Output sequence length should match query sequence length"
        
        # concatinate all attention heads
        att_output = att_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_head*self.dim_head) 

        # final linear transformation to the concatenated output
        att_output = self.output_proj(att_output)

        assert att_output.size() == (batch_size, seq_len, self.dim_model), \
        f"Final output shape {att_output.size()} incorrect"

        return att_output

if __name__ == "__main__":
    torch.manual_seed(42)

    # Hyperparameters (small-scale for quick testing)
    batch_size = 2
    seq_len    = 16
    dim_model    = 128   # num_head * dim_head
    num_head   = 4
    dim_kv     = 64    # KV compression dim  (< dim_model)
    dim_query  = 64    # Q  compression dim  (< dim_model)
    dim_rope   = 32    # RoPE dim

    model = MultiHeadLatentAttention(
        dim_model=dim_model,
        num_head=num_head,
        dim_kv=dim_kv,
        dim_query=dim_query,
        dim_rope=dim_rope,
        dropout=0.0,
        bias=True,
        max_batch_size=batch_size,
        max_seq_len=128,
    )
    model.eval()

    x = torch.randn(batch_size, seq_len, dim_model)

    # --- 1. Standard forward (no cache) ---
    with torch.no_grad():
        out = model(x)
    print(f"[self-attn, no cache]  input: {x.shape}  output: {out.shape}")
    assert out.shape == (batch_size, seq_len, dim_model), "Shape mismatch!"

    # --- 2. Cached inference (prefill then decode) ---
    model2 = MultiHeadLatentAttention(
        dim_model=dim_model,
        num_head=num_head,
        dim_kv=dim_kv,
        dim_query=dim_query,
        dim_rope=dim_rope,
        dropout=0.0,
        bias=True,
        max_batch_size=batch_size,
        max_seq_len=128,
    )
    model2.eval()

    # Prefill: feed the whole prompt at once
    prompt = torch.randn(batch_size, seq_len, dim_model)
    with torch.no_grad():
        out_prefill = model2(prompt, use_cache=True, start_pos=0)
    print(f"[cached, prefill]      input: {prompt.shape}  output: {out_prefill.shape}")

    # Decode: generate one token at a time
    for step in range(4):
        token = torch.randn(batch_size, 1, dim_model)
        with torch.no_grad():
            out_decode = model2(token, use_cache=True, start_pos=seq_len + step)
        print(f"[cached, decode step {step}] input: {token.shape}  output: {out_decode.shape}")

    print("\nAll assertions passed. MLA forward OK!")