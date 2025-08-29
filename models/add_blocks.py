import torch
import torch.nn as nn
import torch.nn.functional as F

class FirstConvBlock(nn.Module):
    """
    Basic convolutional block.
    Consists of a convolutional layer, a max pooling layer and a dropout layer.
    """
    def __init__(
        self,
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        pool_size: int, 
        dropout: float
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        self.mp = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, in_channels, seq_len)
        x = F.relu(self.conv(x))  # (batch_size, out_channels, seq_len)
        x = self.mp(x)  # (batch_size, out_channels, seq_len // pool_size)
        x = self.do(x)  # (batch_size, out_channels, seq_len // pool_size)
        return x
    
class SELayerSimple(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            reduction: int=4
        ):
        super().__init__()
        self.fc = nn.Sequential(
                nn.Linear(out_channels, int(in_channels // reduction)),
                nn.SiLU(),
                nn.Linear(int(in_channels // reduction), out_channels),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y

class SwiGLULayer(nn.Module):
    def __init__(self, dim):
        super(SwiGLULayer, self).__init__()
        self.dim = dim
        self.swish = nn.SiLU() # same as swish

    def forward(self, x):
        out, gate = torch.chunk(x, 2, dim = self.dim)
        return out * self.swish(gate)


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, embedding_dim, mult=4, rate = 0.0, use_bias = True):
        super(FeedForwardSwiGLU, self).__init__()
        swiglu_out = int(embedding_dim * mult/2)
        self.layernorm = nn.LayerNorm(embedding_dim,eps = 1e-6)
        self.linear1 = nn.Linear(embedding_dim,embedding_dim * mult, bias = use_bias)
        self.swiglulayer = SwiGLULayer(dim = 1)
        self.drop = nn.Dropout(rate)
        self.linear2 = nn.Linear(swiglu_out,embedding_dim, bias = use_bias)

    def forward(self, inputs):
        x = self.layernorm(inputs.transpose(1,2)) # Swap dimensions and make channel dim=2
        x = self.linear1(x) 
        x = self.swiglulayer(x.transpose(1,2)) # Swap dimensions again and make channel dim =1
        x = self.drop(x)
        x = self.linear2(x.transpose(1,2)) # Swap dimensions and make channel dim=2
        out = self.drop(x.transpose(1,2)) # Swap dimensions again and make channel dim =1
        return out

class ConformerSASwiGLULayer(nn.Module):
    def __init__(self, embedding_dim,  ff_mult = 4, kernel_size = 15, rate = 0.2, num_heads = 4, use_bias = False):
        super(ConformerSASwiGLULayer, self).__init__()
        self.ff1 = FeedForwardSwiGLU(embedding_dim = embedding_dim, mult = ff_mult, rate = rate, use_bias = use_bias)
        self.layernorm1 = nn.LayerNorm(embedding_dim,eps = 1e-6)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=kernel_size, groups=embedding_dim, padding='same', bias = False),
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1, bias = True),
            nn.ReLU(),
            nn.Dropout(rate),
        )
        self.layernorm2 = nn.LayerNorm(embedding_dim,eps = 1e-6)    
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first = True)
        self.ff2 = FeedForwardSwiGLU(embedding_dim = embedding_dim, mult = ff_mult, rate = rate, use_bias = use_bias)

    def forward(self, x):
        x = x.float()
        x = x + 0.5 * self.ff1(x)
        
        x1 = x.transpose(1,2)
        x1 = self.layernorm1(x1) #channel dim = 2
        x1 = x1.transpose(1, 2)
        x1 = x1 + self.conv(x1)
        
        x = x + x1
        x = x.transpose(1, 2) # output channel dim = 2
        x = self.layernorm2(x)
        x = x + self.attn(x, x, x)[0]
        x = x.transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        
        return x