import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from collections import OrderedDict
from . import blocks

class DREAM_RNN(nn.Module):
    def __init__(
        self, 
        in_channels: int = 4,
        first_out_channels: int = 320,
        core_out_channels: int = 320,
        final_block: str = 'human',
        seqsize: int = 200,
        lstm_hidden_channels: int = 320,
        first_kernel_sizes: List[int] = [9, 15],
        core_kernel_sizes: List[int] = [9, 15],
        pool_size: int = 1,
        first_dropout: float = 0.2,
        core_dropout_1: float = 0.2,
        core_dropout_2: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        each_first_out_channels = first_out_channels // len(first_kernel_sizes)
        self.conv_list_first = nn.ModuleList([
            blocks.FirstConvBlock(in_channels, each_first_out_channels, k, pool_size, first_dropout) for k in first_kernel_sizes
        ])

        # core block
        each_core_out_channels = core_out_channels // len(core_kernel_sizes)
        self.lstm = nn.LSTM(input_size=first_out_channels, 
                            hidden_size=lstm_hidden_channels, 
                            batch_first=True, 
                            bidirectional=True)
        self.conv_list_core = nn.ModuleList([
            blocks.FirstConvBlock(2 * lstm_hidden_channels, each_core_out_channels, k, pool_size, core_dropout_1) for k in core_kernel_sizes
        ])
        self.do = nn.Dropout(core_dropout_2)

        # final block
        if final_block == 'human':
            self.final_block = blocks.HumanFinalBlock(in_channels=core_out_channels)
        elif final_block == 'yeast':
            self.final_block = blocks.YeastFinalBlock(in_channels=core_out_channels)
        else:
            raise ValueError("Final block must be either 'human' or 'yeast'")
    
    def forward(self, x) -> torch.Tensor:
        # x: (batch_size, 4, seq_len), 4 channels: A, C, G, T
        if len(x.shape) < 3:
            x = F.one_hot(x.to(torch.int64), self.in_channels)
            x = x.float().permute(0,2,1)

        # get the output of each convolutional layer
        conv_outputs_first = [conv(x) for conv in self.conv_list_first]  # [(batch_size, each_out_channels, seq_len // pool_size), ...]

        # concatenate the outputs along the channel dimension
        x = torch.cat(conv_outputs_first, dim=1)  # (batch_size, out_channels, seq_len // pool_size)

        # core block
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, in_channels)
        x, _ = self.lstm(x)  # (batch_size, seq_len, 2 * lstm_hidden_channels)
        x = x.permute(0, 2, 1)  # (batch_size, 2 * lstm_hidden_channels, seq_len)
        
        # get the output of each convolutional layer
        conv_outputs_core = [conv(x) for conv in self.conv_list_core]  # [(batch_size, each_conv_out_channels, seq_len // pool_size), ...]

        # concatenate the outputs along the channel dimension
        x = torch.cat(conv_outputs_core, dim=1)  # (batch_size, conv_out_channels, seq_len // pool_size)

        x = self.do(x)  # (batch_size, conv_out_channels, seq_len // pool_size)

        # final block
        x = self.final_block(x)
        return x
    
    def predict(self,x):
        x=self(x)
        if isinstance(self.final_block, blocks.HumanFinalBlock):
            return x
        else:
            bins=torch.arange(18,device=self.device)
            x = F.softmax(x, dim=1)
            score = (x * bins).sum(dim=1)
            return score
    
class DREAM_CNN(nn.Module):
    def __init__(
        self, 
        in_channels: int = 4,
        first_out_channels: int = 320,
        core_out_channels: int = 64,
        final_block: str = 'human',
        seqsize: int = 200,
        first_kernel_sizes: List[int] = [9, 15],
        pool_size: int = 1,
        first_dropout: float = 0.2,
        core_dropout: float = 0.1,
        core_resize_factor: int = 4,
        core_se_reduction: int = 4,
        core_bn_momentum: float = .1,
        core_filter_per_group: int = 2,
        core_activation=nn.SiLU,
        core_ks: int = 7,
        core_block_sizes = [128, 128, 64, 64, 64],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        each_first_out_channels = first_out_channels // len(first_kernel_sizes)
        self.conv_list_first = nn.ModuleList([
            blocks.FirstConvBlock(in_channels, each_first_out_channels, k, pool_size, first_dropout) for k in first_kernel_sizes
        ])

        # core block
        seqextblocks = OrderedDict()
        self.core_block_sizes = [first_out_channels] + core_block_sizes + [core_out_channels]
        for ind, (prev_sz, sz) in enumerate(zip(self.core_block_sizes[:-1], self.core_block_sizes[1:])):
            block = nn.Sequential(
                nn.Conv1d(
                    in_channels=prev_sz,
                    out_channels=sz * core_resize_factor,
                    kernel_size=1,
                    padding='same',
                    bias=False
                ),
                nn.BatchNorm1d(sz * core_resize_factor, 
                                momentum=core_bn_momentum),
                core_activation(),
                nn.Dropout(core_dropout),
                
                nn.Conv1d(
                    in_channels=sz * core_resize_factor,
                    out_channels=sz * core_resize_factor,
                    kernel_size=core_ks,
                    groups=sz * core_resize_factor // core_filter_per_group,
                    padding='same',
                    bias=False
                ),
                nn.BatchNorm1d(sz * core_resize_factor, 
                                momentum=core_bn_momentum),
                core_activation(),
                nn.Dropout(core_dropout),
                blocks.SELayerSimple(prev_sz, sz * core_resize_factor, reduction=core_se_reduction),
                nn.Conv1d(
                    in_channels=sz * core_resize_factor,
                    out_channels=prev_sz,
                    kernel_size=1,
                    padding='same',
                    bias=False
                ),
                nn.BatchNorm1d(prev_sz,
                                momentum=core_bn_momentum),
                core_activation(),
                nn.Dropout(core_dropout),
            
            )
            seqextblocks[f'inv_res_blc{ind}'] = block

            block = nn.Sequential(
                nn.Conv1d(
                    in_channels=2 * prev_sz,
                    out_channels=sz,
                    kernel_size=core_ks,
                    padding='same',
                    bias=False
                ),
                nn.BatchNorm1d(sz, 
                                momentum=core_bn_momentum),
                core_activation(),
                nn.Dropout(core_dropout),
            )
            seqextblocks[f'resize_blc{ind}'] = block

        self.seqextractor = nn.ModuleDict(seqextblocks)

        # final block
        if final_block == 'human':
            self.final_block = blocks.HumanFinalBlock(in_channels=core_out_channels)
        elif final_block == 'yeast':
            self.final_block = blocks.YeastFinalBlock(in_channels=core_out_channels)
        else:
            raise ValueError("Final block must be either 'human' or 'yeast'")
    
    def forward(self, x) -> torch.Tensor:
        # x: (batch_size, 4, seq_len), 4 channels: A, C, G, T
        if len(x.shape) < 3:
            x = F.one_hot(x.to(torch.int64), self.in_channels)
            x = x.float().permute(0,2,1)

        # get the output of each convolutional layer
        conv_outputs_first = [conv(x) for conv in self.conv_list_first]  # [(batch_size, each_out_channels, seq_len // pool_size), ...]

        # concatenate the outputs along the channel dimension
        x = torch.cat(conv_outputs_first, dim=1)  # (batch_size, out_channels, seq_len // pool_size)

        # core block
        for i in range(len(self.core_block_sizes) - 1):
            x = torch.cat([x, self.seqextractor[f'inv_res_blc{i}'](x)], dim=1)
            x = self.seqextractor[f'resize_blc{i}'](x)

        # final block
        x = self.final_block(x)
        return x
    
    def predict(self,x):
        x=self(x)
        if isinstance(self.final_block, blocks.HumanFinalBlock):
            return x
        else:
            bins=torch.arange(18,device=self.device)
            x = F.softmax(x, dim=1)
            score = (x * bins).sum(dim=1)
            return score
    
class DREAM_ATTN(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        first_out_channels: int = 256,
        core_out_channels: int=256,
        final_block: str = 'human',
        seqsize: int = 200,
        first_ks: int = 7,
        first_activation = nn.SiLU,
        first_dropout: float = 0.1,
        first_bn_momentum: float = 0.1,
        core_num_heads: int = 8,
        core_ks: int = 15,
        core_dropout: float = 0.1,
        core_n_blocks: int = 4,        
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seqsize = seqsize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # first block
        self.first_block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=first_out_channels,
                kernel_size=first_ks,
                padding='same',
                bias=False
            ),
            nn.BatchNorm1d(first_out_channels,
                            momentum=first_bn_momentum),
            first_activation(),
            nn.Dropout(first_dropout)
        )

        # core block
        self.core_blocks = nn.ModuleList([blocks.ConformerSASwiGLULayer(
                                embedding_dim = first_out_channels,
                                kernel_size = core_ks, rate = core_dropout, 
                                num_heads = core_num_heads) 
                                for _ in range(core_n_blocks)])
        self.core_n_blocks = core_n_blocks
        self.core_out_channels = core_out_channels
        self.core_pos_embedding = nn.Embedding(seqsize, core_out_channels)

        # final block
        if final_block == 'human':
            self.final_block = blocks.HumanFinalBlock(in_channels=core_out_channels)
        elif final_block == 'yeast':
            self.final_block = blocks.YeastFinalBlock(in_channels=core_out_channels)
        else:
            raise ValueError("Final block must be either 'human' or 'yeast'")
        

    def forward(self, x) -> torch.Tensor:
        if len(x.shape) < 3:
            x = F.one_hot(x.to(torch.int64), self.in_channels)
            x = x.float().permute(0,2,1)
        x = self.first_block(x)

        # core block
        x = x.transpose(1,2)

        pos = torch.arange(start=0, end = self.seqsize, step=1).to(self.device)
        pos = pos.unsqueeze(0)
        pos = self.core_pos_embedding(pos.long())
        x = x + pos
        x = x.transpose(1,2)

        for i in range(self.core_n_blocks) :
            x = self.core_blocks[i](x)

        # final block
        x = self.final_block(x)
        return x
    
    def predict(self,x):
        x=self(x)
        if isinstance(self.final_block, blocks.HumanFinalBlock):
            return x
        else:
            bins=torch.arange(18,device=self.device)
            x = F.softmax(x, dim=1)
            score = (x * bins).sum(dim=1)
            return score