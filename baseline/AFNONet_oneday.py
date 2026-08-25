import torch.nn as nn
import torch
from functools import partial

from models.core.AdaptiveFourierNeuralBlock import AdaptiveFourierNeuralBlock
from models.core.PatchEmbedding import PatchEmbedding

class AFNONet (nn.Module) :
    def __init__(
            self,
            input_dim =(720, 1440),
            hidden_dim = 512,
            variables=46,
            debugger=0,
            dtype=torch.float32
    ) :
        super().__init__()

        self.debugger = debugger
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.cube_patch = PatchEmbedding(
            img_size=input_dim,
            patch_size = (8, 8),
            in_channels=variables,
            embedding_dim=hidden_dim,
            dtype=dtype
        )

        self.core_size = (input_dim[0] // 8, input_dim [1] // 8)
        num_patches = self.core_size[0] * self.core_size[1]

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim, dtype=dtype))

        self.encoder = nn.Sequential(* [AdaptiveFourierNeuralBlock(
            dim=hidden_dim,
            input_size=self.core_size,
            dtype=dtype
        ) for i in range(12)])

        self.norm = partial(nn.LayerNorm, eps=1e-6)(hidden_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, variables * 16, kernel_size=(2, 2), stride=(2, 2) ),
            nn.Tanh(),
            nn.ConvTranspose2d(variables * 16, variables * 4, kernel_size=(2, 2), stride=(2, 2)),
            nn.Tanh(),
            nn.ConvTranspose2d(variables * 4, variables, kernel_size=(2, 2), stride=(2, 2))
        )

        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
    
    def forward(self, x) :
        # Batch, Sequence, Layer, Height, Weight
        B, S, L, H, W = x.shape  # [1, 1, 46, 720, 1440] B,T,C,H,W

        assert S == 1, "AIGOMS model use only at 1 sequence as input"

        x = x.view(B, L, H, W)

        x = self.cube_patch(x)
        x = x + self.pos_embed

        x = self.encoder(x)  # [1, 16200, 16]

        x = self.norm(x).transpose(1, 2)  # [1, 16200, 16] -- [1, 16, 16200]
        x = x.view(-1, self.hidden_dim, * self.core_size)  # [1, 16, 90, 180]

        x = self.decoder(x)  # [1, 46, 720, 1440]

        x = x.view(B, S, L, H, W)

        return x

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total_params * 4 / (1024 ** 2)  # float32 = 4 bytes
    print(f"📊 Total Parameters     : {total_params:,}")
    print(f"✅ Trainable Parameters : {trainable_params:,}")
    print(f"💾 Model Size           : {size_mb:.2f} MB")


if __name__ == "__main__":
    import torch

    # 输入维度定义
    B = 1        # batch size
    S = 1        # sequence length (时间维度)
    L = 46       # number of variables
    H, W = 720, 1440  # spatial resolution
    dtype = torch.float32

    # 创建模型
    model = AFNONet(
        input_dim=(H, W),
        hidden_dim=16,
        variables=L,
        debugger=0,
        dtype=dtype
    )

    # 打印参数信息
    count_parameters(model)

    # 生成随机输入
    x = torch.randn((B, S, L, H, W), dtype=dtype)

    # 将模型和数据移到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = x.to(device)
    model.eval()

    # 执行前向传播
    with torch.no_grad():
        y = model(x)

    # 输出形状信息
    print(f"✅ Input shape  : {x.shape}")
    print(f"✅ Output shape : {y.shape}")
