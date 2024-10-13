import math
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
                
             