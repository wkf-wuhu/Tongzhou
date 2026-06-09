import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import torch.fft


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., dtype=torch.float32):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, dtype=dtype)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, dtype=dtype)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AdativeFourierNeuralOperator(nn.Module):
    def __init__(self, dim, h=14, w=14, num_blocks=4, fno_bias=True, fno_softshrink=0.00):
        super(AdativeFourierNeuralOperator, self).__init__()
        self.hidden_size = dim
        self.h = h
        self.w = w
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0
        self.scale = 0.02
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.relu = nn.ReLU()
        self.bias = nn.Conv2d(self.hidden_size, self.hidden_size, 1)
        self.softshrink = fno_softshrink

    def multiply(self, input, weights):
        return torch.einsum('...bd, bdk->...bk', input, weights)

    def forward(self, x):
        B, H, W, C = x.shape
        bias = self.bias(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')  
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)
        x_real = F.relu(self.multiply(x.real, self.w1[0]) - self.multiply(x.imag, self.w1[1]) + self.b1[0], inplace=True)
        x_imag = F.relu(self.multiply(x.real, self.w1[1]) + self.multiply(x.imag, self.w1[0]) + self.b1[1], inplace=True) 
        x_real = self.multiply(x_real, self.w2[0]) - self.multiply(x_imag, self.w2[1]) + self.b2[0]
        x_imag = self.multiply(x_real, self.w2[1]) + self.multiply(x_imag, self.w2[0]) + self.b2[1]
        x = torch.stack([x_real, x_imag], dim=-1)
        x = F.softshrink(x, lambd=self.softshrink) if self.softshrink else x
        x = torch.view_as_complex(x)  
        x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)
        x = torch.fft.irfft2(x, s=(self.h, self.w), dim=(1,2), norm='ortho')
        return x + bias


class AFNOBlock(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 h=14,
                 w=14,
                 num_blocks=4,
                 fno_bias=True,
                 fno_softshrink=0.00,
                 double_skip=True):
        super(AFNOBlock, self).__init__()
        self.normlayer1 = norm_layer(dim)
        self.filter = AdativeFourierNeuralOperator(dim, h=h, w=w, num_blocks=num_blocks, fno_bias=fno_bias, fno_softshrink=fno_softshrink)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.normlayer2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        f = self.filter(self.normlayer1(x))
        x = x + self.drop_path(f)
        x = x + self.drop_path(self.mlp(self.normlayer2(x)))
        return x
