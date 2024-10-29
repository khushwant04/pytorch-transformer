import math
from turtle import forward
from sympy import Mul
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class InputEmbedding(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        self.d_model = d_model
        self.vocab_size  = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)   


class PostitionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of size (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        
        # Create a matrix of size (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        # Create the div term with the vector shape d_model
        div_term = torch.exp(torch.arange(0, d_model, 2)).float() * (-math.log(10000.0)/ d_model)
        # apply the postional encoding to the even terms
        pe[:, 0::2] = torch.sin(position * div_term)
        # apply the positional encoding to the odd terms
        pe[:, 0::1] = torch.cos(position * div_term)
        # Register the positional encoding as buffer
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x) 
                
class LayerNormalization(nn.Module):
    
    def __init__(self, features, eps: float=10**-6):
        super().__init__()
        self.eps = eps 
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is  a learnable parameters
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim=True) # (batch, seq_len, 1)
        # keep the dimensions for broadcasting
        std = x.std(dim=-1, keepdim=True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small 
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Dropout(d_ff, d_model) # w2 and b2
        
    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) 
                                                     
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.heads = heads # Number of heads 
        # Make sure d_model is divisible by h
        assert d_model % heads == 0, "d_model is not divisible by heads"
        
        self.d_k = d_model // heads # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod    
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # just apply the formula from the paper
        # (batch, heads, seq_len, d_k) --> (batch, heads, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # write a very low value (indicating -inf) to the positions where mask==0
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, heads, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, heads, seq_len, seq_len) --> (batch, heads, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
                
        # (batch, seq_len, d_model) --> (batch, seq_len, heads, d_k) --> (batch, heads, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1,2)
        
        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, heads, seq_len, d_k) --> (batch, seq_len, heads, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous.view(x.shape[0], -1, self.heads * self.d_k)
        
        # Multiply by Wo
        # (batch, heads, seq_len, d_k) --> (batch, seq_len, d_model)
        return self.w_o(x)
                                                         

class ResidualConnection(nn.Module):
    
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
                                                                 
class EncoderBlock(nn.Module):
    
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))    
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x 
    
    